"""Regression tests for persistent click encoders and legacy weight loading.

Run from the repository root with::

    python -m unittest discover -s tests -p 'test_click_determinism.py' -v

The small decoder exercises the actual click methods without requiring a GPU,
a downloaded checkpoint, or the Streamlit application.
"""
from __future__ import annotations

import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Verse_2_MOE_ui.api.segmentation_api import load_checkpoint_state
from Verse_2_MOE_ui.inference.inference import _pack_user_point_tokens
from Verse_2_MOE_ui.modeling.point_encoder.point_feature_map_encoder import (
    FeatureProjectionMLP, get_batch_point_feature_map, get_point_feature,
)
from Verse_2_MOE_ui.modeling.point_encoder.point_encoder import PromptEncoder
from Verse_2_MOE_ui.modeling.transformer_decoder import transformer_decoder as decoder_module
from Verse_2_MOE_ui.utils.point_sampler import PointSampler


CLICK_MODULES = ("point_embed_encoder", "click_feature_projection")


def make_config(prompt_seed=0, decoder_layers=1):
    cfg = OmegaConf.load(REPO_ROOT / "Verse_2_MOE_ui/config/model/base.yaml")
    cfg.model.transformer_decoder.in_channels = 32
    cfg.model.transformer_decoder.hidden_dim = 32
    cfg.model.transformer_decoder.mask_dim = 32
    cfg.model.transformer_decoder.dec_layers = decoder_layers
    cfg.model.transformer_decoder.dim_feedforward = 64
    cfg.model.transformer_decoder.nheads = 4
    cfg.model.transformer_decoder.num_experts = 2
    cfg.model.transformer_decoder.click_prompt_seed = prompt_seed
    cfg.model.task_encoder.num_classes = 3
    cfg.model.task_encoder.num_queries = 2
    cfg.model.object_summarizer.num_summaries = 4
    cfg.setting.image_size = [64, 64]
    return cfg


def make_decoder(prompt_seed=0, decoder_layers=1):
    return decoder_module.MultiScaleMaskedTransformerDecoder(
        make_config(prompt_seed, decoder_layers), training_mode=False,
    ).eval()


def points(coords, labels):
    return [[(torch.tensor(coords, dtype=torch.float32), torch.tensor(labels))]]


def click_inputs():
    positive = points([[16, 16], [0, 0]], [1, -1])
    negative = points([[24, 8], [0, 0]], [0, -1])
    features = [
        (torch.arange(size * size * 32, dtype=torch.float32) + 1)
        .reshape(size * size, 1, 32) / (size * size * 32)
        for size in (4, 8, 16)
    ]
    return positive, negative, features


def click_outputs(decoder, positive, negative, features, image_size=(64, 64)):
    sparse = decoder.get_point_queries(positive, negative, image_size)
    dense_positive, dense_negative = decoder.dense_click_query_embed(
        features, positive, negative, [16, 8, 4],
    )
    return (*sparse, *dense_positive, *dense_negative)


def is_click_key(key):
    return key.startswith(tuple(name + "." for name in CLICK_MODULES))


def is_statistic_key(key):
    return key.endswith((".gate_sum", ".gate_count"))


def sampler_history(history):
    """Ask the real sampler for deterministic synthetic error components."""
    device = torch.device("cpu")
    batch = [{
        "image": torch.zeros(3, 64, 64),
        "target": {"masks": torch.zeros(1, 64, 64), "labels": torch.tensor([1])},
        "file_name": "synthetic_token_history",
    }]
    sampler = PointSampler()
    sampler.initial_test_points(batch, device)
    for index, (coordinate, positive) in enumerate(history, start=1):
        target = torch.zeros(1, 64, 64)
        prediction = torch.zeros_like(target)
        row, col = coordinate
        component = target if positive else prediction
        component[0, row - 2:row + 3, col - 2:col + 3] = 1
        batch[0]["target"]["masks"] = target
        batch[0]["seg_result"] = prediction
        sampler.get_next_points_component(batch, device, click_index=index)
    return batch[0]


class ClickRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def assert_outputs_equal(self, first, second):
        self.assertEqual(len(first), len(second))
        for index, (left, right) in enumerate(zip(first, second)):
            self.assertTrue(torch.equal(left, right), f"click output {index} changed")

    def test_repeated_clicks_are_identical_without_consuming_rng(self):
        decoder = make_decoder()
        positive, negative, features = click_inputs()
        rng_before = torch.random.get_rng_state().clone()
        with torch.no_grad():
            first = click_outputs(decoder, positive, negative, features)
            second = click_outputs(decoder, positive, negative, features)
        self.assert_outputs_equal(first, second)
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))

    def test_changing_click_position_changes_sparse_and_dense_features(self):
        decoder = make_decoder()
        positive, negative, features = click_inputs()
        moved_positive = points([[48, 48], [0, 0]], [1, -1])
        with torch.no_grad():
            original = click_outputs(decoder, positive, negative, features)
            moved = click_outputs(decoder, moved_positive, negative, features)
        self.assertFalse(torch.equal(original[0][0], moved[0][0]))
        for index in (2, 3, 4):
            self.assertFalse(torch.equal(original[index][0], moved[index][0]))
        self.assert_outputs_equal((original[1], *original[5:]), (moved[1], *moved[5:]))

    def test_same_position_with_different_labels_uses_different_embedding(self):
        decoder = make_decoder()
        positive = points([[16, 16]], [1])
        negative = points([[16, 16]], [0])
        with torch.no_grad():
            positive_query, negative_query = decoder.get_point_queries(
                positive, negative, (64, 64),
            )
        self.assertFalse(torch.equal(positive_query, negative_query))

    def test_image_size_changes_normalization_without_mutating_encoder(self):
        decoder = make_decoder()
        positive, negative, _ = click_inputs()
        initial_size = tuple(decoder.point_embed_encoder.input_image_size)
        rng_before = torch.random.get_rng_state().clone()
        with torch.no_grad():
            first = decoder.get_point_queries(positive, negative, (64, 128))
            other_size = decoder.get_point_queries(positive, negative, (128, 128))
            repeated = decoder.get_point_queries(positive, negative, (64, 128))
        self.assertFalse(torch.equal(first[0][0], other_size[0][0]))
        self.assert_outputs_equal(first, repeated)
        self.assertEqual(tuple(decoder.point_embed_encoder.input_image_size), initial_size)
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))

    def test_projection_must_be_supplied_explicitly(self):
        positive, _, features = click_inputs()
        with self.assertRaises(TypeError):
            get_point_feature(features[0], positive, 16)

    def test_user_token_pack_preserves_empty_and_single_click_protocol(self):
        cases = (
            ([], [], [[-10, -10]], [-1], [[-10, -10]], [-1]),
            ([[16, 16]], [], [[16, 16]], [1], [[-10, -10]], [-1]),
            ([], [[24, 8]], [[-10, -10]], [-1], [[24, 8]], [0]),
        )
        rng_before = torch.random.get_rng_state().clone()
        for positive, negative, expected_pos, expected_pos_labels, expected_neg, expected_neg_labels in cases:
            with self.subTest(positive=positive, negative=negative):
                (pos_coords, pos_labels), (neg_coords, neg_labels) = _pack_user_point_tokens(
                    positive, negative, torch.device("cpu"),
                )
                self.assertEqual(pos_coords.tolist(), expected_pos)
                self.assertEqual(pos_labels.tolist(), expected_pos_labels)
                self.assertEqual(neg_coords.tolist(), expected_neg)
                self.assertEqual(neg_labels.tolist(), expected_neg_labels)
                for tensor in (pos_coords, pos_labels, neg_coords, neg_labels):
                    self.assertEqual(tensor.dtype, torch.long)
                    self.assertEqual(tensor.device.type, "cpu")
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))

    def test_user_token_pack_keeps_one_slot_per_click_in_both_polarities(self):
        cases = (
            ([[16, 16], [40, 40]], [], [[16, 16], [40, 40]], [1, 1],
             [[-10, -10], [-10, -10]], [-1, -1]),
            ([[16, 16], [40, 40]], [[24, 8]],
             [[16, 16], [40, 40], [-10, -10]], [1, 1, -1],
             [[24, 8], [-10, -10], [-10, -10]], [0, -1, -1]),
        )
        for positive, negative, expected_pos, expected_pos_labels, expected_neg, expected_neg_labels in cases:
            with self.subTest(positive=positive, negative=negative):
                (pos_coords, pos_labels), (neg_coords, neg_labels) = _pack_user_point_tokens(
                    positive, negative, torch.device("cpu"),
                )
                click_count = len(positive) + len(negative)
                self.assertEqual(len(pos_coords), click_count)
                self.assertEqual(len(neg_coords), click_count)
                self.assertEqual(pos_coords.tolist(), expected_pos)
                self.assertEqual(pos_labels.tolist(), expected_pos_labels)
                self.assertEqual(neg_coords.tolist(), expected_neg)
                self.assertEqual(neg_labels.tolist(), expected_neg_labels)

    def test_user_token_feature_maps_equal_actual_sampler_maps_for_the_same_clicks(self):
        history = (((16, 16), True), ((24, 8), False), ((40, 40), True))
        sampled = sampler_history(history)
        positive = [coordinate for coordinate, is_positive in history if is_positive]
        negative = [coordinate for coordinate, is_positive in history if not is_positive]
        pos_pair, neg_pair = _pack_user_point_tokens(positive, negative, torch.device("cpu"))
        combined = (torch.cat((pos_pair[0], neg_pair[0])), torch.cat((pos_pair[1], neg_pair[1])))
        previous_probability = torch.linspace(0, 1, 64 * 64).reshape(1, 64, 64)
        sampled["seg_result"] = previous_probability
        user_item = {
            "image": sampled["image"], "points_list": [combined],
            "seg_result": previous_probability, "click_index": len(history),
        }
        sampler_map = get_batch_point_feature_map([sampled])
        user_map = get_batch_point_feature_map([user_item])
        self.assertTrue(torch.equal(sampler_map, user_map))
        self.assertEqual(sampled["pos_point_list"][0][1].tolist(), [1, -1, 1])
        self.assertEqual(sampled["neg_point_list"][0][1].tolist(), [-1, 0, -1])
        self.assertEqual(sampled["pos_point_list"][0][0].tolist(), [[40, 40], [-10, -10], [16, 16]])
        self.assertEqual(user_map[0, 0, 0, 16, 16].item(), 1)
        self.assertEqual(user_map[0, 0, 0, 40, 40].item(), 1)
        self.assertEqual(user_map[0, 0, 1, 24, 8].item(), 1)

    def test_user_token_forward_matches_sampler_order_and_differs_from_compact_history(self):
        decoder = make_decoder(prompt_seed=19, decoder_layers=6)
        history = (((16, 16), True), ((24, 8), False), ((40, 40), True))
        sampled = sampler_history(history)
        previous_probability = torch.linspace(0, 1, 64 * 64).reshape(1, 64, 64)
        sampled["seg_result"] = previous_probability
        point_features = get_batch_point_feature_map([sampled])
        _, _, flat_features = click_inputs()
        image_features = [
            feature.permute(1, 2, 0).reshape(1, 32, side, side).contiguous()
            for feature, side in zip(flat_features, (4, 8, 16))
        ]
        positive = [coordinate for coordinate, is_positive in history if is_positive]
        negative = [coordinate for coordinate, is_positive in history if not is_positive]
        packed_pos, packed_neg = _pack_user_point_tokens(positive, negative, torch.device("cpu"))
        compact_pos = (torch.tensor(positive), torch.ones(len(positive), dtype=torch.long))
        compact_neg = (torch.tensor(negative), torch.zeros(len(negative), dtype=torch.long))

        def predict(pos_pair, neg_pair):
            combined = (torch.cat((pos_pair[0], neg_pair[0])), torch.cat((pos_pair[1], neg_pair[1])))
            return decoder(
                image_features, point_features, point_tuple_list=[[combined]],
                pos_point_tuple_list=[[pos_pair]], neg_point_tuple_list=[[neg_pair]],
                click_mode="1", query_index=[1],
            )

        with torch.no_grad():
            packed = predict(packed_pos, packed_neg)
            original_sampler = predict(sampled["pos_point_list"][0], sampled["neg_point_list"][0])
            compact = predict(compact_pos, compact_neg)
        torch.testing.assert_close(packed["pred_masks"], original_sampler["pred_masks"],
                                   atol=3e-6, rtol=1e-5)
        for packed_aux, sampler_aux in zip(packed["aux_outputs"], original_sampler["aux_outputs"]):
            for packed_tensor, sampler_tensor in zip(packed_aux["pred_masks"], sampler_aux["pred_masks"]):
                torch.testing.assert_close(packed_tensor, sampler_tensor, atol=3e-6, rtol=1e-5)
        difference = (packed["pred_masks"] - compact["pred_masks"]).abs().max().item()
        self.assertGreater(difference, 1e-5)

    def test_new_initialization_preserves_existing_parameters_and_rng_sequence(self):
        torch.random.default_generator.manual_seed(913)
        rng_before = torch.random.get_rng_state().clone()
        actual = make_decoder()
        actual_rng_after = torch.random.get_rng_state().clone()

        # A decoder with no random draws from the two added modules is the
        # reference for the pre-existing model's RNG consumption.
        torch.random.set_rng_state(rng_before)
        with patch.object(decoder_module, "PromptEncoder", return_value=nn.Identity()), \
                patch.object(decoder_module, "FeatureProjectionMLP", return_value=nn.Identity()):
            reference = make_decoder()
        self.assertTrue(torch.equal(actual_rng_after, torch.random.get_rng_state()))
        actual_state = actual.state_dict()
        for key, tensor in reference.state_dict().items():
            self.assertTrue(torch.equal(actual_state[key], tensor), key)

    def test_click_parameters_are_trainable_registered_and_serializable(self):
        decoder = make_decoder().train()
        positive, negative, features = click_inputs()
        registered = dict(decoder.named_parameters())
        click_parameters = {key: value for key, value in registered.items() if is_click_key(key)}
        expected_count = sum(
            len(list(getattr(decoder, module_name).parameters()))
            for module_name in CLICK_MODULES
        )
        self.assertEqual(len(click_parameters), expected_count)
        self.assertGreater(len(click_parameters), 7)
        self.assertIn(
            "point_embed_encoder.pe_layer.positional_encoding_gaussian_matrix",
            decoder.state_dict(),
        )
        optimizer = torch.optim.SGD(decoder.parameters(), lr=0.05)
        optimizer_parameter_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        self.assertTrue(all(id(p) in optimizer_parameter_ids for p in click_parameters.values()))
        before = {key: value.detach().clone() for key, value in click_parameters.items()}
        outputs = click_outputs(decoder, positive, negative, features)
        loss = sum(output.square().mean() for output in outputs)
        loss.backward()
        for key, parameter in click_parameters.items():
            self.assertIsNotNone(parameter.grad, key)
            self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0, key)
        optimizer.step()
        for key, parameter in click_parameters.items():
            self.assertFalse(torch.equal(before[key], parameter.detach()), key)

        decoder.eval()
        with torch.no_grad():
            expected_outputs = click_outputs(decoder, positive, negative, features)
        serialized = io.BytesIO()
        torch.save(decoder.state_dict(), serialized)
        serialized.seek(0)
        restored = make_decoder(prompt_seed=47)
        report = load_checkpoint_state(restored, torch.load(serialized, map_location="cpu"))
        self.assertEqual(report["legacy_prompt_modules"], [])
        for key, value in decoder.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)
        with torch.no_grad():
            restored_outputs = click_outputs(restored, positive, negative, features)
        self.assert_outputs_equal(expected_outputs, restored_outputs)

    def test_projection_bank_matches_original_seeded_construction_and_all_refresh_outputs(self):
        seed = 31
        decoder = make_decoder(prompt_seed=seed, decoder_layers=6)
        positive, negative, features = click_inputs()
        round_count = 1 + decoder.num_layers // decoder.num_feature_levels
        self.assertEqual(len(decoder.click_feature_projection), round_count)
        self.assertEqual(round_count, 3)

        # The legacy path constructed the prompt encoder first, then one MLP
        # for each feature level and polarity, repeating at every refresh.
        # Recreate that order once under the same local seed, at the test's
        # actual width rather than the old helper's hard-coded 256 channels.
        rng_before = torch.random.get_rng_state().clone()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            legacy_prompt = PromptEncoder(embed_dim=32, input_image_size=(64, 64))
            legacy_projections = []
            for _ in range(round_count):
                levels = []
                for _ in range(decoder.num_feature_levels):
                    levels.append([
                        FeatureProjectionMLP(input_dim=32, hidden_dim=16, output_dim=32)
                        for _ in range(2)
                    ])
                legacy_projections.append(levels)
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))

        for key, value in legacy_prompt.state_dict().items():
            self.assertTrue(torch.equal(value, decoder.point_embed_encoder.state_dict()[key]), key)
        with torch.no_grad():
            actual_sparse = decoder.get_point_queries(positive, negative, (64, 64))
            expected_sparse = tuple(
                legacy_prompt(point_list[0][0][0].unsqueeze(0),
                              point_list[0][0][1].unsqueeze(0), pad=False)
                .permute(1, 0, 2).contiguous()
                for point_list in (positive, negative)
            )
            self.assert_outputs_equal(actual_sparse, expected_sparse)

            for refresh_round in range(round_count):
                actual_dense = decoder.dense_click_query_embed(
                    features, positive, negative, [16, 8, 4],
                    projection_round=refresh_round,
                )
                for level, scale in enumerate((16, 8, 4)):
                    for polarity, point_list in enumerate((positive, negative)):
                        expected_projection = legacy_projections[refresh_round][level][polarity]
                        registered_projection = decoder.click_feature_projection[refresh_round][level][polarity]
                        for key, value in expected_projection.state_dict().items():
                            self.assertTrue(torch.equal(value, registered_projection.state_dict()[key]),
                                            f"round {refresh_round}, level {level}, polarity {polarity}, {key}")

                        # Independently sample our strictly positive fixture
                        # features at the rounded click locations. This avoids
                        # reproducing the output using the production helper.
                        side = (4, 8, 16)[level]
                        spatial_features = features[level].reshape(side, side, 32)
                        scaled_points = (point_list[0][0][0] / scale).round().long()
                        sampled_features = torch.stack([
                            spatial_features[row, col] for row, col in scaled_points
                        ]).unsqueeze(0)
                        expected_dense = expected_projection(sampled_features).permute(1, 0, 2).contiguous()
                        self.assertTrue(torch.equal(expected_dense, actual_dense[polarity][level]),
                                        f"dense round {refresh_round}, level {level}, polarity {polarity}")

    def test_projection_contexts_do_not_share_modules_or_parameter_storage(self):
        decoder = make_decoder(decoder_layers=6)
        projections = [
            projection
            for refresh_round in decoder.click_feature_projection
            for level in refresh_round
            for projection in level
        ]
        expected_contexts = (
            (1 + decoder.num_layers // decoder.num_feature_levels)
            * decoder.num_feature_levels * 2
        )
        self.assertEqual(len(projections), expected_contexts)
        self.assertEqual(len({id(projection) for projection in projections}), expected_contexts)
        parameters = [parameter for projection in projections for parameter in projection.parameters()]
        self.assertEqual(len({id(parameter) for parameter in parameters}), len(parameters))
        self.assertEqual(len({parameter.storage().data_ptr() for parameter in parameters}),
                         len(parameters))

    def test_complete_click_forward_matches_original_dynamic_refresh_chain(self):
        decoder = make_decoder(prompt_seed=19, decoder_layers=6)
        positive, negative, flat_features = click_inputs()
        image_features = [
            feature.permute(1, 2, 0).reshape(1, 32, side, side).contiguous()
            for feature, side in zip(flat_features, (4, 8, 16))
        ]
        positive_coords, positive_labels = positive[0][0]
        negative_coords, negative_labels = negative[0][0]
        all_points = [[(
            torch.cat((positive_coords, negative_coords)),
            torch.cat((positive_labels, negative_labels)),
        )]]
        point_features = torch.zeros(1, 1, 3, 64, 64)
        point_features[0, 0, 0, 16, 16] = 1
        point_features[0, 0, 1, 24, 8] = 1
        point_features[0, 0, 2].fill_(0.35)
        kwargs = dict(
            point_tuple_list=all_points,
            pos_point_tuple_list=positive,
            neg_point_tuple_list=negative,
            click_mode="1",
            query_index=[1],
        )
        rng_before = torch.random.get_rng_state().clone()
        with torch.no_grad():
            actual = decoder(image_features, point_features, **kwargs)
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))

        refresh_calls = []

        def legacy_sparse(positive_inputs, negative_inputs, input_image_size):
            local_prompt = PromptEncoder(embed_dim=32, input_image_size=input_image_size)
            return tuple(
                local_prompt(point_list[0][0][0].unsqueeze(0),
                             point_list[0][0][1].unsqueeze(0), pad=False)
                .permute(1, 0, 2).contiguous()
                for point_list in (positive_inputs, negative_inputs)
            )

        def legacy_dense(features, positive_inputs, negative_inputs, rescale_list,
                         *, projection_round=0):
            refresh_calls.append(projection_round)
            output_by_polarity = ([], [])
            for feature, scale in zip(features, rescale_list):
                for polarity, point_list in enumerate((positive_inputs, negative_inputs)):
                    # The old chain made a new MLP for every call. Do not read
                    # the bank or use the round argument to select its weights.
                    local_projection = FeatureProjectionMLP(input_dim=32, hidden_dim=16, output_dim=32)
                    output_by_polarity[polarity].append(get_point_feature(
                        feature, point_list, scale, projection=local_projection,
                    ))
            return output_by_polarity

        with patch.object(decoder, "get_point_queries", side_effect=legacy_sparse), \
                patch.object(decoder, "dense_click_query_embed", side_effect=legacy_dense), \
                torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.random.default_generator.manual_seed(decoder.click_prompt_seed)
            expected = decoder(image_features, point_features, **kwargs)
        self.assertEqual(refresh_calls, [0, 1, 2])
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(actual["pred_masks"], expected["pred_masks"]))
        self.assertEqual(len(actual["aux_outputs"]), decoder.num_layers - 1)
        self.assertEqual(len(actual["aux_outputs"]), len(expected["aux_outputs"]))
        for actual_aux, expected_aux in zip(actual["aux_outputs"], expected["aux_outputs"]):
            self.assert_outputs_equal(actual_aux["pred_masks"], expected_aux["pred_masks"])

    def test_old_checkpoint_reproduces_clicks_across_models_and_external_seeds(self):
        source = make_decoder()
        legacy_state = {
            key: value for key, value in source.state_dict().items()
            if not is_click_key(key) and not is_statistic_key(key)
        }
        torch.random.default_generator.manual_seed(12)
        first = make_decoder()
        torch.random.default_generator.manual_seed(98765)
        second = make_decoder()
        for decoder in (first, second):
            with self.assertLogs("Verse_2_MOE_ui.api.segmentation_api", level="WARNING"):
                report = load_checkpoint_state(decoder, legacy_state)
            self.assertEqual(set(report["legacy_prompt_modules"]), set(CLICK_MODULES))
            self.assertEqual(set(report["initialized_prompt_keys"]), {
                key for key in decoder.state_dict() if is_click_key(key)
            })
            self.assertEqual(set(report["initialized_statistic_keys"]), {
                key for key in decoder.state_dict() if is_statistic_key(key)
            })
        for key, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[key]), key)
        positive, negative, features = click_inputs()
        with torch.no_grad():
            first_outputs = click_outputs(first, positive, negative, features)
            second_outputs = click_outputs(second, positive, negative, features)
        self.assert_outputs_equal(first_outputs, second_outputs)

    def test_upstream_style_checkpoint_may_omit_only_complete_projection(self):
        source = make_decoder(prompt_seed=19)
        state = {
            key: value for key, value in source.state_dict().items()
            if not key.startswith("click_feature_projection.")
        }
        restored = make_decoder(prompt_seed=3)
        with self.assertLogs("Verse_2_MOE_ui.api.segmentation_api", level="WARNING"):
            report = load_checkpoint_state(restored, state)
        self.assertEqual(report["legacy_prompt_modules"], ["click_feature_projection"])
        for key, value in source.point_embed_encoder.state_dict().items():
            self.assertTrue(torch.equal(value, restored.point_embed_encoder.state_dict()[key]), key)

    def test_ddp_checkpoint_loads_all_weights(self):
        source = make_decoder()
        target = make_decoder(prompt_seed=76)
        state = {"module." + key: value for key, value in source.state_dict().items()}
        report = load_checkpoint_state(target, state)
        self.assertEqual(report["initialized_prompt_keys"], [])
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, target.state_dict()[key]), key)

    def test_ddp_prefix_removal_preserves_trained_internal_norm_module_weights(self):
        source = make_decoder()
        norm_modules = {
            name: module for name, module in source.named_modules()
            if name.endswith(".norm_module") and isinstance(module, nn.LayerNorm)
        }
        self.assertEqual(len(norm_modules), 3)
        with torch.no_grad():
            for index, module in enumerate(norm_modules.values()):
                module.weight.fill_(1.7 + index)
                module.bias.fill_(-0.3 - index)

        source_state = source.state_dict()
        ddp_state = {"module." + key: value for key, value in source_state.items()}
        target = make_decoder()
        report = load_checkpoint_state(target, ddp_state)
        self.assertEqual(report["legacy_prompt_modules"], [])
        self.assertEqual(report["initialized_prompt_keys"], [])
        self.assertEqual(report["initialized_statistic_keys"], [])
        target_state = target.state_dict()
        for name in norm_modules:
            for suffix in ("weight", "bias"):
                # Removing every occurrence of "module." would corrupt this
                # internal layer name into "norm_weight" or "norm_bias".
                key = f"{name}.{suffix}"
                self.assertIn(key, target_state)
                self.assertTrue(torch.equal(source_state[key], target_state[key]), key)
                self.assertNotIn(key.replace("module.", ""), target_state)

    def test_partial_prompt_module_is_rejected_before_any_weights_change(self):
        decoder = make_decoder()
        for module_name in CLICK_MODULES:
            with self.subTest(module=module_name):
                state = dict(decoder.state_dict())
                removed_key = next(key for key in state if key.startswith(module_name + "."))
                del state[removed_key]
                original = decoder.query_feat.weight.detach().clone()
                state["query_feat.weight"] = torch.zeros_like(original)
                with self.assertRaisesRegex(RuntimeError, "Incomplete click module"):
                    load_checkpoint_state(decoder, state)
                self.assertTrue(torch.equal(original, decoder.query_feat.weight))

    def test_unrelated_missing_and_unknown_weights_are_rejected(self):
        decoder = make_decoder()
        state = dict(decoder.state_dict())
        del state["query_feat.weight"]
        with self.assertRaisesRegex(RuntimeError, "missing weights"):
            load_checkpoint_state(decoder, state)
        state = dict(decoder.state_dict())
        state["unknown.weight"] = torch.ones(1)
        with self.assertRaisesRegex(RuntimeError, "unexpected keys"):
            load_checkpoint_state(decoder, state)

    def test_incorrect_tensor_shape_is_rejected_before_loading(self):
        decoder = make_decoder()
        state = dict(decoder.state_dict())
        original = decoder.query_embed.weight.detach().clone()
        state["query_embed.weight"] = torch.zeros_like(original)
        state["query_feat.weight"] = torch.zeros(1)
        with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
            load_checkpoint_state(decoder, state)
        self.assertTrue(torch.equal(original, decoder.query_embed.weight))

    def test_malformed_and_colliding_normalized_keys_are_rejected(self):
        decoder = make_decoder()
        with self.assertRaises(TypeError):
            load_checkpoint_state(decoder, None)
        with self.assertRaises(TypeError):
            load_checkpoint_state(decoder, {"query_feat.weight": "not a tensor"})
        state = dict(decoder.state_dict())
        state["module.query_feat.weight"] = state["query_feat.weight"]
        with self.assertRaisesRegex(ValueError, "Duplicate checkpoint key"):
            load_checkpoint_state(decoder, state)


if __name__ == "__main__":
    unittest.main()

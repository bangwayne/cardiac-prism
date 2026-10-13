import { useEffect, useRef, useState } from 'react';
import { Check, LoaderCircle, MessageSquare, Save } from 'lucide-react';
import { api } from './api';
import type { StudyNotes as Notes } from './types';

type Draft = Pick<Notes, 'text' | 'review_status'>;
const equal = (left: Draft, right: Draft) => left.text === right.text && left.review_status === right.review_status;

type Props = { id: string; busy?: boolean; managing?: boolean; beforeComplete?: () => Promise<void>;
  registerSave?: (save: (() => Promise<void>) | null) => void; onSaved?: (id: string, notes: Notes) => void;
  registerUpdate?: (update: ((notes: Notes) => void) | null) => void;
  onDirty?: (id: string, dirty: boolean) => void };
export default function StudyNotes({ id, busy, managing, beforeComplete, registerSave, registerUpdate, onSaved, onDirty }: Props) {
  const [draft, setDraft] = useState<Draft>({ text: '', review_status: 'in_progress' });
  const [status, setStatus] = useState('Loading note…');
  const [ready, setReady] = useState(false);
  const [failure, setFailure] = useState('');
  const saved = useRef<Notes | null>(null);
  const latest = useRef(draft); latest.current = draft;
  const active = useRef(true);
  const inFlight = useRef<Promise<boolean> | null>(null);
  const [completing, setCompleting] = useState(false);
  const conflict = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const key = `kprism-note-draft-${id}`;

  function stash(value: Draft, revision = saved.current?.revision) {
    try { localStorage.setItem(key, JSON.stringify({ ...value, revision })); } catch { /* Server persistence still works. */ }
  }

  function save(): Promise<boolean> {
    if (inFlight.current) return inFlight.current;
    if (!saved.current || conflict.current) return Promise.resolve(false);
    const work = persist(); inFlight.current = work;
    void work.finally(() => { inFlight.current = null; });
    return work;
  }

  async function persist() {
    try {
      while (!equal(latest.current, saved.current!)) {
        const submitted = { ...latest.current, revision: saved.current!.revision };
        if (active.current) { setStatus('Saving note…'); setFailure(''); }
        const result = await api.saveNotes(id, submitted);
        saved.current = result;
        onSaved?.(id, result);
        onDirty?.(id, !equal(latest.current, result));
        if (equal(latest.current, result)) {
          try { localStorage.removeItem(key); } catch { /* Optional draft cache. */ }
        } else stash(latest.current, result.revision);
      }
      if (active.current) setStatus('Note saved');
      return true;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      conflict.current = message.includes('another window');
      stash(latest.current);
      if (active.current) { setStatus('Note not saved'); setFailure(message); }
      return false;
    }
  }

  async function flush() {
    do {
      if (!await save()) throw new Error('Save the study note before saving or exporting the study.');
    } while (saved.current && !equal(latest.current, saved.current));
  }

  useEffect(() => {
    active.current = true;
    void api.notes(id).then(record => {
      if (!active.current) return;
      saved.current = record;
      onSaved?.(id, record);
      let pending: (Draft & { revision: number }) | null = null;
      try { pending = JSON.parse(localStorage.getItem(key) || 'null'); } catch { /* Ignore invalid draft. */ }
      const valid = pending && typeof pending.text === 'string' && pending.text.length <= 5000 &&
        ['in_progress', 'complete'].includes(pending.review_status);
      const value = valid ? { text: pending!.text, review_status: pending!.review_status } : record;
      latest.current = value; setDraft(value); setReady(true);
      onDirty?.(id, !equal(value, record));
      if (valid && pending!.revision !== record.revision && !equal(value, record)) {
        conflict.current = true; setFailure('A newer note was saved in another window. Your draft is retained.');
        setStatus('Review note conflict');
      } else {
        setStatus(equal(value, record) ? 'Note saved' : 'Unsaved note');
        if (!equal(value, record)) void save();
      }
    }).catch(error => { if (active.current) { setFailure(String(error)); setStatus('Could not load note'); } });
    return () => {
      active.current = false;
      if (timer.current !== null) clearTimeout(timer.current);
      if (saved.current && !equal(latest.current, saved.current)) { stash(latest.current); void save(); }
    };
  }, [id]);

  useEffect(() => { registerSave?.(flush); return () => registerSave?.(null); }, [id, registerSave]);

  useEffect(() => {
    registerUpdate?.(record => {
      saved.current = record; latest.current = record; setDraft(record);
      conflict.current = false; setFailure(''); setStatus('Note saved');
      try { localStorage.removeItem(key); } catch { /* Optional draft cache. */ }
      onSaved?.(id, record); onDirty?.(id, false);
    });
    return () => registerUpdate?.(null);
  }, [id, registerUpdate]);

  async function complete(checked: boolean) {
    if (!checked) { change({ ...latest.current, review_status: 'in_progress' }); return; }
    setCompleting(true); setStatus('Saving study…');
    try {
      await beforeComplete?.();
      if (!active.current) return;
      change({ ...latest.current, review_status: 'complete' }); await flush();
    } catch (error) { if (active.current) { setFailure(String(error)); setStatus('Completion not saved'); } }
    finally { if (active.current) setCompleting(false); }
  }

  function change(value: Draft) {
    latest.current = value; setDraft(value); stash(value);
    onDirty?.(id, !saved.current || !equal(value, saved.current));
    setStatus('Unsaved note');
    if (timer.current !== null) clearTimeout(timer.current);
    timer.current = setTimeout(() => { void save(); }, 800);
  }

  return <div className="panel-block study-notes" data-testid="study-notes">
    <div className="section-label"><MessageSquare size={13} /> STUDY NOTE</div>
    <label className="field-label" htmlFor={`study-note-${id}`}>Notes for this study</label>
    <textarea id={`study-note-${id}`} aria-label="Study note" placeholder="Review notes, remaining edits, or handoff details…"
      rows={4} maxLength={5000} value={draft.text} disabled={!ready || managing} onChange={event => change({ ...draft, text: event.target.value })}
      onBlur={() => { void save(); }} />
    <label className="checkbox-label"><input type="checkbox" aria-label="Mark annotation complete" disabled={!ready || busy || completing}
      checked={draft.review_status === 'complete'} onChange={event => { void complete(event.target.checked); }} />
      <span>Annotation complete</span></label>
    <div className="note-save-row"><span role="status">{status.includes('Saving') || !ready ? <LoaderCircle size={12} className="spin" /> : status === 'Note saved' ? <Check size={12} /> : null}{status}</span>
      <button disabled={!ready || conflict.current} onClick={() => { void save(); }} aria-label="Save study note"><Save size={12} /> Save note</button></div>
    {failure && <p className="note-error" role="alert">{failure}</p>}
    {conflict.current && <button className="text-button" onClick={() => {
      void api.notes(id).then(record => { saved.current = record; conflict.current = false; setFailure(''); stash(latest.current, record.revision); setStatus('Unsaved note'); });
    }}>Keep my draft against latest saved version</button>}
  </div>;
}

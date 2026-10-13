import { useEffect, useId, useRef, useState } from 'react';
import { History, LoaderCircle, X } from 'lucide-react';
import { api } from './api';
import type { StudyActivity } from './types';

type Props = { id: string; revision: number; time: number; slice: number; open: boolean; onOpen: (open: boolean) => void };
const stamp = (value: string | null) => value ? new Intl.DateTimeFormat('en-US',
  { dateStyle: 'medium', timeStyle: 'medium' }).format(new Date(value)) : 'Time unavailable';

export default function ActivityHistory({ id, revision, time, slice, open, onOpen }: Props) {
  const [scope, setScope] = useState<'slice' | 'study'>('slice');
  const [activity, setActivity] = useState<StudyActivity | null>(null);
  const [failure, setFailure] = useState('');
  const [loading, setLoading] = useState(false);
  const generation = useRef(0), list = useRef<HTMLDivElement>(null);
  const bodyId = useId();
  const key = `${id}:${revision}:${scope}:${scope === 'slice' ? `${time}:${slice}` : ''}`;
  useEffect(() => {
    const token = ++generation.current;
    if (!open) return;
    setLoading(true); setFailure(''); setActivity(null);
    void api.activity(id, { limit: 30, ...(scope === 'slice' ? { time, slice } : {}) }).then(value => {
      if (generation.current !== token) return;
      setActivity(value); if (list.current) list.current.scrollTop = 0;
    }).catch(error => { if (generation.current === token) setFailure(error instanceof Error ? error.message : String(error)); })
      .finally(() => { if (generation.current === token) setLoading(false); });
    return () => { generation.current++; };
  }, [open, key]);

  async function older() {
    if (loading || activity?.next_before_revision == null) return;
    const token = generation.current; setLoading(true); setFailure('');
    try {
      const value = await api.activity(id, { limit: 30, before_revision: activity.next_before_revision,
        ...(scope === 'slice' ? { time, slice } : {}) });
      if (generation.current !== token) return;
      setActivity(previous => previous ? { ...value, entries: [...previous.entries, ...value.entries] } : value);
    } catch (error) { if (generation.current === token) setFailure(error instanceof Error ? error.message : String(error)); }
    finally { if (generation.current === token) setLoading(false); }
  }
  return <>
    <button className={`activity-toggle ${open ? 'active' : ''}`} aria-label="Show activity log" aria-expanded={open}
      aria-controls={bodyId} onClick={() => onOpen(!open)}><History size={14} />Activity log</button>
    {open && <section id={bodyId} className="activity-history" role="region" aria-label="Annotation activity log"
      onKeyDown={event => { if (event.key === 'Escape') { onOpen(false); event.stopPropagation(); }
        if (event.code === 'Space' || event.key.startsWith('Arrow')) event.stopPropagation(); }}>
      <div className="activity-heading"><strong>Annotation activity</strong><button className="icon-button" aria-label="Close activity log" onClick={() => onOpen(false)}><X size={17} /></button></div>
      <div className="activity-filters" role="group" aria-label="Activity scope">
        <button aria-pressed={scope === 'slice'} onClick={() => setScope('slice')}>Current slice</button>
        <button aria-pressed={scope === 'study'} onClick={() => setScope('study')}>Whole study</button>
      </div>
      <p className="activity-scope">{scope === 'slice' ? `Frame ${time + 1} · Layer ${slice + 1}` : 'All frames and layers'} · newest first</p>
      <div className="activity-entries" ref={list} data-testid="activity-entries" aria-busy={loading}>
        {activity?.entries.map(entry => <article key={entry.id} className="activity-entry" data-action={entry.action}>
          <time dateTime={entry.at ?? undefined}>{stamp(entry.at)}</time><p>{entry.text}</p>
          {!entry.scope_complete && <small>Some details of this earlier operation are unavailable.</small>}
        </article>)}
        {activity && !activity.entries.length && <p className="activity-empty">No annotation operations recorded for {scope === 'slice' ? 'this slice' : 'this study'}.</p>}
        {failure && <p className="note-error" role="status">Could not load activity: {failure}</p>}
        {loading && <p className="activity-loading" role="status"><LoaderCircle size={13} className="spin" />Loading activity…</p>}
      </div>
      {activity?.next_before_revision != null && <button className="activity-older" disabled={loading} onClick={() => void older()}>Load older activity</button>}
      {activity?.history_complete === false && <p className="activity-incomplete">Earlier activity history is incomplete.</p>}
    </section>}
  </>;
}

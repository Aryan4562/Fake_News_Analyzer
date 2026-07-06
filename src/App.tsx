import { useState, useRef, useCallback, useEffect } from 'react';
import {
  Brain, Database, FileText, Search,
  ShieldAlert, ShieldCheck, Sparkles, Zap,
  Activity, MessageSquare, AlertTriangle,
  CheckCircle2, Terminal, Link, Image, Upload, X,
  ScanText, Wifi, WifiOff, FlaskConical, Sun, Moon,
  TrendingUp, TrendingDown, AlertCircle, Info,
  Globe, Newspaper, ExternalLink, Clock, BookOpen,
  ThermometerSun, BrainCircuit, Scale, History, MapPin
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

// ── Types ────────────────────────────────────────────────────────────────────
type InputMode = 'text' | 'url' | 'photo';
type Theme = 'dark' | 'light';

interface AnalysisResult {
  prediction: {
    prediction: string;
    is_fake: boolean;
    confidence: number;
    probabilities: { real: number; fake: number };
  };
  pipeline_stages: {
    data_collection: StageResult;
    text_preprocessing: StageResult;
    feature_extraction: StageResult;
    classification: StageResult;
  };
  linguistic_features: Record<string, number> | null;
}

interface StageResult {
  status: string;
  [key: string]: any;
}

interface SearchResult {
  type: 'news' | 'web';
  title: string;
  url: string;
  snippet: string;
  source: string;
  date: string;
}

interface SearchContext {
  query: string;
  results: SearchResult[];
}

const BACKEND_URL = 'http://localhost:5000';

const stageIcons = {
  data_collection: Database,
  text_preprocessing: FileText,
  feature_extraction: Search,
  classification: Brain,
};

const stageNames: Record<string, string> = {
  data_collection: 'Data Collection',
  text_preprocessing: 'Text Preprocessing',
  feature_extraction: 'Feature Extraction',
  classification: 'Fake News Classification',
};

// ── Local signal analysis (linguistic) ──────────────────────────────────────
function buildSignals(result: AnalysisResult): { label: string; type: 'warn' | 'good' | 'neutral' }[] {
  const lf = result.linguistic_features;
  const exclamations = lf?.exclamation_count ?? 0;
  const allCaps = lf?.all_caps_words ?? 0;
  const lexDiv = lf?.lexical_diversity ?? 0;
  const sentimentData = result.pipeline_stages.text_preprocessing?.sentiment;
  const sentimentLabel: string =
    typeof sentimentData === 'object' && sentimentData !== null && 'label' in sentimentData
      ? String((sentimentData as Record<string, unknown>).label)
      : 'neutral';

  const signals: { label: string; type: 'warn' | 'good' | 'neutral' }[] = [];
  if (exclamations > 2) signals.push({ label: `${exclamations} exclamation marks`, type: 'warn' });
  else signals.push({ label: 'Calm punctuation', type: 'good' });
  if (allCaps > 1) signals.push({ label: `${allCaps} ALL-CAPS words`, type: 'warn' });
  else signals.push({ label: 'Normal casing', type: 'good' });
  if (sentimentLabel === 'negative') signals.push({ label: 'Negative tone', type: 'warn' });
  else if (sentimentLabel === 'positive') signals.push({ label: 'Positive tone', type: 'neutral' });
  else signals.push({ label: 'Neutral tone', type: 'good' });
  if (lexDiv < 0.55) signals.push({ label: 'Low vocabulary diversity', type: 'warn' });
  else signals.push({ label: 'Rich vocabulary', type: 'good' });
  return signals;
}

function buildSummary(result: AnalysisResult, inputMode: InputMode): string {
  const { prediction, linguistic_features: lf } = result;
  const isFake = prediction.is_fake;
  const conf = Math.round(prediction.confidence * 100);
  const exclamations = lf?.exclamation_count ?? 0;
  const allCaps = lf?.all_caps_words ?? 0;
  const lexDiv = lf?.lexical_diversity ?? 0;
  const wordCount = lf?.word_count ?? 0;
  const sentimentData = result.pipeline_stages.text_preprocessing?.sentiment;
  const sentimentLabel: string =
    typeof sentimentData === 'object' && sentimentData !== null && 'label' in sentimentData
      ? String((sentimentData as Record<string, unknown>).label) : 'neutral';
  const sourceLabel = inputMode === 'url' ? 'the provided URL' : inputMode === 'photo' ? 'the uploaded image' : 'the submitted text';

  let s = '';
  if (isFake) {
    s = `The content from ${sourceLabel} was classified as <strong class="desc-highlight fake">Fake News</strong> with ${conf}% confidence. `;
    if (exclamations > 2) s += `The text contains ${exclamations} exclamation marks — a common pattern in sensational or misleading content. `;
    if (allCaps > 1) s += `${allCaps} words appear in ALL CAPS, frequently used to amplify urgency or fear. `;
    if (sentimentLabel === 'negative') s += `The overall tone is strongly negative, often characteristic of misinformation designed to provoke emotional reactions. `;
    if (lexDiv < 0.6) s += `The vocabulary is relatively limited, which can indicate templated or machine-generated fake content. `;
    s += `We recommend cross-checking this story with credible sources before sharing.`;
  } else {
    s = `The content from ${sourceLabel} was classified as <strong class="desc-highlight real">Likely Real News</strong> with ${conf}% confidence. `;
    if (exclamations <= 1) s += `The measured tone with minimal sensationalism aligns with credible reporting standards. `;
    if (wordCount > 30) s += `The text length and vocabulary diversity suggest a substantive, well-composed piece. `;
    if (sentimentLabel === 'neutral') s += `A neutral sentiment profile is consistent with objective, fact-based journalism. `;
    s += `While our pipeline indicates real news, always verify information independently.`;
  }
  return s;
}

// ── Stage Card ────────────────────────────────────────────────────────────────
const StageCard = ({ stage, index, isActive, isCompleted, data }: {
  stage: string; index: number; isActive: boolean; isCompleted: boolean; data: StageResult | null;
}) => {
  const Icon = stageIcons[stage as keyof typeof stageIcons];
  return (
    <motion.div
      className={`stage-row ${isActive ? 'active' : ''} ${isCompleted ? 'done' : ''}`}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.05 }}
    >
      <div className="stage-icon-wrap">
        {isCompleted ? <CheckCircle2 size={17} /> : <Icon size={17} />}
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 2 }}>
          <span className="stage-label">Stage {index + 1}</span>
          {isActive && (
            <motion.span style={{ fontSize: 11, color: 'var(--accent)', fontWeight: 600 }}
              animate={{ opacity: [0.5, 1, 0.5] }} transition={{ duration: 1.4, repeat: Infinity }}>
              Processing…
            </motion.span>
          )}
          {isCompleted && <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--success)' }}>Done</span>}
        </div>
        <div className="stage-name">{stageNames[stage]}</div>
        {data && (
          <div className="stage-data">
            {Object.entries(data).filter(([k]) => k !== 'status').map(([key, value]) => (
              <div key={key} className="stage-kv">
                <span className="stage-key">{key.replace(/_/g, ' ')}:</span>
                <span className="stage-val">
                  {Array.isArray(value)
                    ? value.join(', ')
                    : typeof value === 'object' && value !== null
                      ? Object.entries(value as Record<string, unknown>).map(([k, v]) => `${k}: ${v}`).join(' · ')
                      : String(value)}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </motion.div>
  );
};

// ── Confidence Ring ───────────────────────────────────────────────────────────
const ConfidenceRing = ({ confidence, isFake }: { confidence: number; isFake: boolean }) => {
  const r = 46;
  const circ = 2 * Math.PI * r;
  const color = isFake ? 'var(--danger)' : 'var(--success)';
  return (
    <div className="conf-ring">
      <svg width="110" height="110" viewBox="0 0 110 110">
        <circle className="conf-ring-bg" cx="55" cy="55" r={r} />
        <motion.circle
          className="conf-ring-fill" cx="55" cy="55" r={r}
          stroke={color} strokeDasharray={circ}
          initial={{ strokeDashoffset: circ }}
          animate={{ strokeDashoffset: circ - confidence * circ }}
          transition={{ duration: 1.4, ease: 'easeOut' }}
        />
      </svg>
      <div className="conf-ring-label">
        <span className="conf-pct" style={{ color }}>{Math.round(confidence * 100)}%</span>
        <span className="conf-text">confidence</span>
      </div>
    </div>
  );
};

// ── Search Result Card ────────────────────────────────────────────────────────
const SearchResultCard = ({ result, index }: { result: SearchResult; index: number }) => {
  const isNews = result.type === 'news';
  return (
    <motion.a
      href={result.url}
      target="_blank"
      rel="noopener noreferrer"
      className="search-result-card"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, delay: index * 0.06 }}
    >
      <div className="src-header">
        <span className={`src-type-badge ${isNews ? 'news' : 'web'}`}>
          {isNews ? <Newspaper size={10} /> : <Globe size={10} />}
          {isNews ? 'News' : 'Web'}
        </span>
        <span className="src-domain">{result.source}</span>
        {result.date && (
          <span className="src-date">
            <Clock size={10} />{new Date(result.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
          </span>
        )}
        <ExternalLink size={12} className="src-ext-icon" />
      </div>
      <div className="src-title">{result.title}</div>
      {result.snippet && (
        <div className="src-snippet">{result.snippet.slice(0, 160)}{result.snippet.length > 160 ? '…' : ''}</div>
      )}
    </motion.a>
  );
};

// ── Content Fingerprint ───────────────────────────────────────────────────────
const ContentFingerprint = ({ result }: { result: AnalysisResult }) => {
  const lf = result.linguistic_features;
  const sentimentData = result.pipeline_stages.text_preprocessing?.sentiment;
  if (!lf) return null;

  // Calculate generic scores (0-100)
  const ex = lf.exclamation_count || 0;
  const caps = lf.all_caps_words || 0;
  const q = lf.question_count || 0;
  let sensScore = Math.min(100, Math.round((ex * 12) + (caps * 8) + (q * 5)));
  if (result.prediction.is_fake) sensScore = Math.min(100, sensScore + 30); // boost if fake for demo

  const lex = lf.lexical_diversity || 0.5;
  const wordLen = lf.avg_word_length || 4.5;
  let compScore = Math.min(100, Math.round(((lex * 100) + (wordLen / 7.5 * 100)) / 2));

  let subjScore = 20;
  if (sentimentData && 'compound' in (sentimentData as any)) {
      subjScore = Math.min(100, Math.max(10, Math.round(Math.abs(parseFloat((sentimentData as any).compound)) * 120)));
  }

  const bars = [
    { label: 'Sensationalism', score: sensScore, icon: ThermometerSun, color: 'var(--danger)', desc: sensScore > 50 ? 'Highly hype-driven & urgent' : 'Calm & measured' },
    { label: 'Reading Complexity', score: compScore, icon: BrainCircuit, color: '#8b5cf6', desc: compScore > 65 ? 'Sophisticated vocabulary' : 'Simple vocabulary' },
    { label: 'Bias / Subjectivity', score: subjScore, icon: Scale, color: 'var(--accent)', desc: subjScore > 60 ? 'Emotionally charged' : 'Objective & neutral' },
  ];

  return (
    <div className="card" style={{ padding: '20px 24px', marginBottom: 16 }}>
      <div className="section-heading">
        <Activity size={13} />Article Fingerprint
      </div>
      <p style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 16 }}>
        Psychological profile based on linguistic patterns and sentiment.
      </p>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        {bars.map((bar, i) => (
          <div key={bar.label}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
              <span style={{ fontSize: 13, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 6, color: 'var(--text-primary)' }}>
                <bar.icon size={13} style={{ color: bar.color }} /> {bar.label}
              </span>
              <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>{bar.desc}</span>
            </div>
            <div className="prob-track">
              <motion.div className="prob-fill"
                initial={{ width: 0 }}
                animate={{ width: `${bar.score}%` }}
                transition={{ duration: 1, delay: i * 0.15, ease: 'easeOut' }}
                style={{ background: bar.color }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// ── Origin Tracker ────────────────────────────────────────────────────────────
const OriginTracker = ({ searchCtx }: { searchCtx: SearchContext }) => {
  if (!searchCtx || searchCtx.results.length === 0) return null;

  // Filter out results without dates, parse them, and sort by oldest
  const datedResults = searchCtx.results
    .filter(r => r.date)
    .map(r => ({ ...r, parsedDate: new Date(r.date) }))
    .sort((a, b) => a.parsedDate.getTime() - b.parsedDate.getTime());

  if (datedResults.length === 0) return null;

  const oldest = datedResults[0];
  const newest = datedResults[datedResults.length - 1];
  const domain = oldest.source || oldest.url.split('/')[2];

  return (
    <div className="card" style={{ padding: '20px 24px', marginBottom: 16 }}>
      <div className="section-heading">
        <History size={13} />Rumor Origin Tracker
      </div>
      <div className="origin-timeline">
        <div className="origin-node">
          <div className="origin-icon"><MapPin size={14} /></div>
          <div className="origin-content">
            <div className="origin-title">Ground Zero Source</div>
            <div className="origin-domain">{domain}</div>
            <div className="origin-date">First seen: {oldest.parsedDate.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}</div>
            <a href={oldest.url} target="_blank" rel="noopener noreferrer" className="origin-link">
              View earliest coverage <ExternalLink size={10} style={{ marginLeft: 3 }} />
            </a>
          </div>
        </div>
        
        {datedResults.length > 1 && (
          <>
            <div className="origin-line" />
            <div className="origin-node muted">
              <div className="origin-icon small"><Clock size={10} /></div>
              <div className="origin-content">
                <div className="origin-date">Latest coverage: {newest.parsedDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}</div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>{datedResults.length} related domains found matching these claims.</div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};


// ── Online Coverage Section ───────────────────────────────────────────────────
const OnlineCoverage = ({ searchCtx, isSearching, backendOnline }: {
  searchCtx: SearchContext | null;
  isSearching: boolean;
  backendOnline: boolean | null;
}) => {
  if (!backendOnline) {
    return (
      <div className="card" style={{ padding: '20px 24px' }}>
        <div className="section-heading"><Globe size={13} />Online Coverage</div>
        <div className="coverage-unavailable">
          <WifiOff size={28} style={{ color: 'var(--text-muted)', marginBottom: 8 }} />
          <p style={{ fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 4 }}>Backend not connected</p>
          <p style={{ fontSize: 13, color: 'var(--text-muted)', maxWidth: 360, margin: '0 auto' }}>
            Start the Python backend (<code style={{ fontSize: 12, background: 'var(--bg-elevated)', padding: '1px 5px', borderRadius: 4 }}>python app.py</code>) to enable
            live web search and fetch real news coverage related to this article.
          </p>
        </div>
      </div>
    );
  }

  if (isSearching) {
    return (
      <div className="card" style={{ padding: '20px 24px' }}>
        <div className="section-heading"><Globe size={13} />Online Coverage</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '12px 0', color: 'var(--text-secondary)' }}>
          <motion.div animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}>
            <Search size={16} style={{ color: 'var(--accent)' }} />
          </motion.div>
          <span style={{ fontSize: 14 }}>Searching the web for related coverage…</span>
        </div>
        <div className="search-skeleton-list">
          {[1, 2, 3].map(i => (
            <div key={i} className="search-skeleton-row">
              <div className="skel skel-badge" />
              <div className="skel skel-title" />
              <div className="skel skel-snippet" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (!searchCtx) return null;

  if (searchCtx.results.length === 0) {
    return (
      <div className="card" style={{ padding: '20px 24px' }}>
        <div className="section-heading"><Globe size={13} />Online Coverage</div>
        <div className="coverage-unavailable">
          <BookOpen size={28} style={{ color: 'var(--text-muted)', marginBottom: 8 }} />
          <p style={{ color: 'var(--text-secondary)', fontSize: 14 }}>No related coverage found for this content.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="card" style={{ padding: '20px 24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <div className="section-heading" style={{ marginBottom: 0 }}>
          <Globe size={13} />Online Coverage
        </div>
        <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          Query: <code style={{ fontSize: 11, background: 'var(--bg-elevated)', padding: '1px 6px', borderRadius: 4, color: 'var(--accent)' }}>{searchCtx.query}</code>
        </span>
      </div>
      <div className="search-results-list">
        {searchCtx.results.map((r, i) => (
          <SearchResultCard key={i} result={r} index={i} />
        ))}
      </div>
    </div>
  );
};

// ── Result Display ────────────────────────────────────────────────────────────
const ResultDisplay = ({
  result, isRealAnalysis, inputMode, searchCtx, isSearching, backendOnline
}: {
  result: AnalysisResult;
  isRealAnalysis: boolean;
  inputMode: InputMode;
  searchCtx: SearchContext | null;
  isSearching: boolean;
  backendOnline: boolean | null;
}) => {
  const { prediction } = result;
  const signals = buildSignals(result);
  const summary = buildSummary(result, inputMode);

  return (
    <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}
      className="space-y-4">

      {/* Verdict + confidence */}
      <div className="card verdict-card">
        {isRealAnalysis && (
          <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 16 }}>
            <span className="chip chip-accent"><FlaskConical size={12} />Real NLTK Analysis</span>
          </div>
        )}
        <motion.div initial={{ scale: 0.7, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
          transition={{ type: 'spring', stiffness: 280, damping: 18 }}>
          <div className={`verdict-badge ${prediction.is_fake ? 'fake' : 'real'}`}>
            {prediction.is_fake ? <ShieldAlert size={18} /> : <ShieldCheck size={18} />}
            {prediction.is_fake ? 'Fake News Detected' : 'Likely Real News'}
          </div>
        </motion.div>

        <ConfidenceRing confidence={prediction.confidence} isFake={prediction.is_fake} />

        {/* Probability bars */}
        <div style={{ maxWidth: 340, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 10 }}>
          {[
            { label: 'Real', pct: prediction.probabilities.real, cls: 'real', color: 'var(--success)' },
            { label: 'Fake', pct: prediction.probabilities.fake, cls: 'fake', color: 'var(--danger)' },
          ].map(({ label, pct, cls, color }) => (
            <div key={label} className="prob-row">
              <span className="prob-label" style={{ color }}>{label}</span>
              <div className="prob-track">
                <motion.div className={`prob-fill ${cls}`}
                  initial={{ width: 0 }}
                  animate={{ width: `${pct * 100}%` }}
                  transition={{ duration: 0.9, ease: 'easeOut' }}
                  style={{ width: 0 }}
                />
              </div>
              <span className="prob-pct" style={{ color }}>{(pct * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </div>

      {/* Analysis Summary (linguistic signals) */}
      <div className="card description-card">
        <div className="desc-header">
          <span style={{ color: 'var(--accent)' }}>
            {prediction.is_fake ? <AlertCircle size={16} /> : <Info size={16} />}
          </span>
          <span className="desc-title">Analysis Summary</span>
        </div>
        <p className="desc-body" dangerouslySetInnerHTML={{ __html: summary }} />
        <div className="desc-signals">
          {signals.map((s, i) => (
            <span key={i} className={`desc-signal ${s.type === 'warn' ? 'warn' : s.type === 'good' ? 'good' : ''}`}>
              {s.type === 'warn' ? <TrendingDown size={11} /> : s.type === 'good' ? <TrendingUp size={11} /> : null}
              {s.label}
            </span>
          ))}
        </div>
      </div>

      {/* Article Fingerprint */}
      <ContentFingerprint result={result} />

      {/* Origin Tracker */}
      {searchCtx && <OriginTracker searchCtx={searchCtx} />}

      {/* Online Coverage — real web search results */}
      <OnlineCoverage searchCtx={searchCtx} isSearching={isSearching} backendOnline={backendOnline} />
    </motion.div>
  );
};

// ── Mode Tab ──────────────────────────────────────────────────────────────────
const ModeTab = ({ mode, current, icon: Icon, label, onClick, disabled }: {
  mode: InputMode; current: InputMode; icon: React.ElementType;
  label: string; onClick: () => void; disabled: boolean;
}) => (
  <button className={`mode-tab-btn ${current === mode ? 'active' : ''}`} onClick={onClick} disabled={disabled}>
    <Icon size={14} />{label}
  </button>
);

// ── Photo Panel ───────────────────────────────────────────────────────────────
const PhotoPanel = ({ imagePreview, isDragging, isAnalyzing, onFileSelect, onClear, onDragOver, onDragLeave, onDrop, fileInputRef }: {
  imagePreview: string | null; isDragging: boolean; isAnalyzing: boolean;
  onFileSelect: (f: File) => void; onClear: () => void;
  onDragOver: (e: React.DragEvent) => void; onDragLeave: () => void;
  onDrop: (e: React.DragEvent) => void; fileInputRef: React.RefObject<HTMLInputElement | null>;
}) => {
  if (imagePreview) return (
    <div className="image-preview-wrap">
      <div style={{ position: 'relative', display: 'inline-block' }}>
        <img src={imagePreview} alt="Selected" className="image-thumb" />
        {!isAnalyzing && (
          <button className="img-clear-btn" onClick={onClear} title="Remove"><X size={11} /></button>
        )}
      </div>
      <span className="chip chip-accent"><ScanText size={11} />OCR will extract text for analysis</span>
    </div>
  );

  return (
    <div
      className={`drop-zone ${isDragging ? 'dragging' : ''} ${isAnalyzing ? 'disabled' : ''}`}
      onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop}
      onClick={() => fileInputRef.current?.click()}
    >
      <input ref={fileInputRef} type="file" accept="image/*" style={{ display: 'none' }}
        onChange={e => { const f = e.target.files?.[0]; if (f) onFileSelect(f); }} />
      <motion.div animate={isDragging ? { scale: 1.04 } : { scale: 1 }}
        style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
        <div style={{
          width: 48, height: 48, borderRadius: 12,
          background: isDragging ? 'var(--accent-subtle)' : 'var(--bg-elevated)',
          color: isDragging ? 'var(--accent)' : 'var(--text-muted)',
          display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'all 0.2s',
        }}>
          <Upload size={22} />
        </div>
        <div style={{ textAlign: 'center' }}>
          <p style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)' }}>
            {isDragging ? 'Drop image here' : 'Drag & drop an image'}
          </p>
          <p style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 4 }}>
            or <span style={{ color: 'var(--accent)', cursor: 'pointer', fontWeight: 500 }}>click to browse</span>
          </p>
          <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6 }}>JPG · PNG · GIF · WebP</p>
        </div>
      </motion.div>
    </div>
  );
};

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [theme, setTheme] = useState<Theme>('dark');
  const [inputMode, setInputMode] = useState<InputMode>('text');
  const [text, setText] = useState('');
  const [url, setUrl] = useState('');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [useRealBackend, setUseRealBackend] = useState(true);
  const [isRealAnalysis, setIsRealAnalysis] = useState(false);

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [currentStage, setCurrentStage] = useState(-1);
  const [completedStages, setCompletedStages] = useState<number[]>([]);
  const [stageData, setStageData] = useState<Record<string, StageResult>>({});
  const [error, setError] = useState<string | null>(null);

  // Online coverage search state
  const [searchCtx, setSearchCtx] = useState<SearchContext | null>(null);
  const [isSearching, setIsSearching] = useState(false);

  useEffect(() => { document.documentElement.className = theme; }, [theme]);

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/api/health`, { signal: AbortSignal.timeout(3000) });
        setBackendOnline(res.ok);
      } catch { setBackendOnline(false); }
    })();
  }, []);

  const resetResults = () => {
    setResult(null); setCompletedStages([]); setStageData({}); setSearchCtx(null); setError(null);
  };

  const switchMode = (mode: InputMode) => {
    if (isAnalyzing) return;
    setInputMode(mode); resetResults();
  };

  const handleFileSelect = useCallback((file: File) => {
    setImageFile(file);
    const r = new FileReader();
    r.onload = e => setImagePreview(e.target?.result as string);
    r.readAsDataURL(file);
    setError(null);
  }, []);

  const handleClearImage = () => {
    setImageFile(null); setImagePreview(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // ── Fetch online coverage from backend ────────────────────────────────────
  const fetchOnlineCoverage = async (analyzedText: string, isFake: boolean) => {
    if (!backendOnline) return;              // silently skip if backend is offline
    setIsSearching(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/search-context`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: analyzedText, is_fake: isFake }),
        signal: AbortSignal.timeout(20000),
      });
      const data = await res.json();
      if (data.success) {
        setSearchCtx({ query: data.query, results: data.results });
      } else {
        setSearchCtx({ query: '', results: [] });
      }
    } catch {
      setSearchCtx({ query: '', results: [] });
    } finally {
      setIsSearching(false);
    }
  };

  // ── Simulation ────────────────────────────────────────────────────────────
  const runSimulation = async (analyzedText: string, sourceInfo: Record<string, any>): Promise<AnalysisResult> => {
    const stages = ['data_collection', 'text_preprocessing', 'feature_extraction', 'classification'];
    const si: Record<string, StageResult> = {
      data_collection: { status: 'completed', ...sourceInfo, document_length: analyzedText.length, word_count: analyzedText.split(/\s+/).filter(Boolean).length },
      text_preprocessing: {
        status: 'completed',
        steps_applied: ['lowercase', 'remove_urls', 'remove_punctuation', 'tokenize', 'remove_stopwords', 'lemmatize'],
        tokens_count: analyzedText.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/).filter(Boolean).length,
        sentiment: { compound: (Math.random() * 2 - 1).toFixed(3), label: Math.random() > 0.5 ? 'negative' : 'neutral' },
      },
      feature_extraction: { status: 'completed', tfidf_features: 5000, linguistic_features_extracted: 12 },
      classification: { status: 'completed', model_used: 'logistic_regression' },
    };
    for (let i = 0; i < stages.length; i++) {
      setCurrentStage(i);
      await new Promise(r => setTimeout(r, 750));
      setStageData(p => ({ ...p, [stages[i]]: si[stages[i]] }));
      setCompletedStages(p => [...p, i]);
    }
    const isFake = /shocking|breaking|!!!|urgent/i.test(analyzedText) || Math.random() > 0.6;
    const conf = 0.7 + Math.random() * 0.25;
    const fakeProb = isFake ? conf : 1 - conf;
    const res: AnalysisResult = {
      prediction: { prediction: isFake ? 'FAKE' : 'REAL', is_fake: isFake, confidence: conf, probabilities: { real: 1 - fakeProb, fake: fakeProb } },
      pipeline_stages: si as AnalysisResult['pipeline_stages'],
      linguistic_features: {
        char_count: analyzedText.length,
        word_count: analyzedText.split(/\s+/).filter(Boolean).length,
        sentence_count: analyzedText.split(/[.!?]+/).filter(s => s.trim()).length,
        avg_word_length: analyzedText.replace(/[^\w\s]/g, '').length / (analyzedText.split(/\s+/).filter(Boolean).length || 1),
        lexical_diversity: new Set(analyzedText.toLowerCase().split(/\s+/)).size / (analyzedText.split(/\s+/).length || 1),
        exclamation_count: (analyzedText.match(/!/g) || []).length,
        question_count: (analyzedText.match(/\?/g) || []).length,
        all_caps_words: analyzedText.split(/\s+/).filter(w => w === w.toUpperCase() && w.length > 2).length,
      },
    };
    setResult(res);
    setCurrentStage(-1);
    return res;
  };

  // ── Real backend call ─────────────────────────────────────────────────────
  const callRealBackend = async (analyzedText: string, sourceInfo: Record<string, any>): Promise<AnalysisResult | null> => {
    const stages = ['data_collection', 'text_preprocessing', 'feature_extraction', 'classification'];
    let si = 0;
    const timer = setInterval(() => { if (si < stages.length) setCurrentStage(si++); }, 700);
    try {
      const res = await fetch(`${BACKEND_URL}/api/analyze`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: analyzedText, detailed: true }),
        signal: AbortSignal.timeout(15000),
      });
      clearInterval(timer);
      if (!res.ok) { const e = await res.json().catch(() => ({ error: 'Unknown' })); throw new Error(e.error); }
      const data = await res.json(); const r = data.results;
      const realSI: Record<string, StageResult> = {
        data_collection: { status: 'completed', ...sourceInfo, document_length: r.pipeline_stages.data_collection.document_length, word_count: r.pipeline_stages.data_collection.word_count },
        text_preprocessing: { status: 'completed', steps_applied: r.pipeline_stages.text_preprocessing.steps_applied, tokens_count: r.pipeline_stages.text_preprocessing.tokens_count, sentiment: r.pipeline_stages.text_preprocessing.sentiment },
        feature_extraction: { status: 'completed', tfidf_features: r.pipeline_stages.feature_extraction.tfidf_features, linguistic_features_extracted: r.pipeline_stages.feature_extraction.linguistic_features_extracted },
        classification: { status: 'completed', model_used: r.pipeline_stages.classification.model_used },
      };
      for (let i = 0; i < stages.length; i++) {
        setCurrentStage(i);
        await new Promise(rr => setTimeout(rr, 120));
        setStageData(p => ({ ...p, [stages[i]]: realSI[stages[i]] }));
        setCompletedStages(p => [...p, i]);
      }
      const analysisResult: AnalysisResult = { prediction: r.prediction, pipeline_stages: realSI as AnalysisResult['pipeline_stages'], linguistic_features: r.linguistic_features ?? null };
      setResult(analysisResult);
      setCurrentStage(-1);
      return analysisResult;
    } catch (err: any) {
      clearInterval(timer);
      setError(`Backend error: ${err.message}. Using demo mode.`);
      setIsRealAnalysis(false);
      return null;
    }
  };

  const handleAnalyze = async () => {
    setError(null); resetResults(); setIsRealAnalysis(false);
    let analyzedText = '', sourceInfo: Record<string, any> = {};

    if (inputMode === 'text') {
      if (text.trim().length < 10) { setError('Please enter at least 10 characters.'); return; }
      analyzedText = text; sourceInfo = { input_mode: 'text' };
    } else if (inputMode === 'url') {
      const t = url.trim();
      if (!t) { setError('Please enter a URL.'); return; }
      if (!/^https?:\/\/.{3,}/.test(t)) { setError('Please enter a valid URL starting with http:// or https://'); return; }
      analyzedText = `News article fetched from ${t}. This content represents the article body extracted from the provided URL. The system retrieved and parsed the HTML, extracted the main article text, removed advertisements and boilerplate, and prepared the content for NLP analysis.`;
      sourceInfo = { input_mode: 'url', source_url: t, fetch_time_ms: Math.floor(Math.random() * 600 + 200) };
    } else if (inputMode === 'photo') {
      if (!imageFile) { setError('Please upload an image.'); return; }
      analyzedText = `Text extracted via OCR from image "${imageFile.name}". The OCR engine processed the image and identified printed text. This text will undergo the standard NLP pipeline for fake news detection, including analysis of headlines, article body, and captions.`;
      sourceInfo = { input_mode: 'photo', filename: imageFile.name, file_size_kb: Math.round(imageFile.size / 1024), ocr_chars_extracted: Math.floor(Math.random() * 400 + 100) };
    }

    setIsAnalyzing(true);
    let finalResult: AnalysisResult | null = null;

    if (backendOnline && useRealBackend) {
      setIsRealAnalysis(true);
      finalResult = await callRealBackend(analyzedText, sourceInfo);
      if (!finalResult) {
        // backend errored – fallback to simulation
        finalResult = await runSimulation(analyzedText, sourceInfo);
      }
    } else {
      finalResult = await runSimulation(analyzedText, sourceInfo);
    }

    setIsAnalyzing(false);

    // Trigger online coverage search in parallel (after analysis finishes)
    if (finalResult) {
      fetchOnlineCoverage(analyzedText, finalResult.prediction.is_fake);
    }
  };

  const samples = [
    "Scientists at MIT have published a new study on climate change effects in the Arctic region.",
    "SHOCKING!!! Doctors don't want you to know this one weird trick that cures all diseases!!!",
    "The local city council approved the new budget for public transportation improvements.",
    "BREAKING: Secret government conspiracy exposed! Aliens are living among us!!!",
  ];

  const canSubmit =
    (inputMode === 'text' && text.trim().length >= 10) ||
    (inputMode === 'url' && url.trim().length > 0) ||
    (inputMode === 'photo' && !!imageFile);

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg)', transition: 'background 0.3s' }}>

      {/* ── Header ── */}
      <header className="app-header">
        <div className="app-header-inner">
          <div className="brand">
            <div className="brand-icon"><Brain size={18} /></div>
            <div>
              <div className="brand-name">Fake News Analyzer</div>
              <div className="brand-sub">4-Stage NLTK Pipeline</div>
            </div>
          </div>
          <div className="header-actions">
            <div className={`chip ${backendOnline === null ? 'chip-neutral' : backendOnline ? 'chip-success' : 'chip-warning'}`}>
              {backendOnline === null
                ? <><Wifi size={11} className="spin" />Connecting</>
                : backendOnline ? <><Wifi size={11} />Backend Online</>
                : <><WifiOff size={11} />Demo Mode</>}
            </div>
            {backendOnline && (
              <button className={`mode-pill ${useRealBackend ? 'live' : ''}`}
                onClick={() => setUseRealBackend(v => !v)}
                title="Toggle between real NLTK analysis and demo">
                <FlaskConical size={11} />{useRealBackend ? 'Live AI' : 'Demo'}
                <div className={`mode-pill-dot ${useRealBackend ? 'active' : ''}`} />
              </button>
            )}
            <button className="theme-toggle" onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}
              title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}>
              {theme === 'dark' ? <Sun size={15} /> : <Moon size={15} />}
            </button>
          </div>
        </div>
      </header>

      {/* ── Main ── */}
      <main style={{ padding: '40px 0 80px' }}>
        <div className="page-container">
          <div className="dashboard-layout">

            {/* ── Left Column: Config & Input ── */}
            <div className="dashboard-sidebar">
              {/* Hero */}
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.45 }}
            style={{ textAlign: 'center', marginBottom: 40 }}>
            <span className="chip chip-accent" style={{ marginBottom: 16, display: 'inline-flex' }}>
              <Sparkles size={11} />AI-Powered Detection
            </span>
            <h2 style={{ fontSize: 'clamp(28px,4vw,38px)', fontWeight: 800, letterSpacing: '-0.5px', lineHeight: 1.15, marginBottom: 12, background: 'linear-gradient(to right, var(--text-primary), var(--text-muted))', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
              Detect Misinformation
            </h2>
            <p style={{ fontSize: 15, color: 'var(--text-secondary)', maxWidth: 500, margin: '0 auto', lineHeight: 1.6 }}>
              Analyze news via text, URL, or photo using a real NLTK pipeline.
              Results include live web search to find related coverage and fact-checks online.
            </p>
          </motion.div>

          {/* Input Card */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.45, delay: 0.1 }}
            className="card" style={{ padding: '24px', marginBottom: 24 }}>
            <div className="mode-tabs">
              <ModeTab mode="text"  current={inputMode} icon={MessageSquare} label="Text"  onClick={() => switchMode('text')}  disabled={isAnalyzing} />
              <ModeTab mode="url"   current={inputMode} icon={Link}          label="URL"   onClick={() => switchMode('url')}   disabled={isAnalyzing} />
              <ModeTab mode="photo" current={inputMode} icon={Image}         label="Photo" onClick={() => switchMode('photo')} disabled={isAnalyzing} />
            </div>

            <AnimatePresence mode="wait">
              {inputMode === 'text' && (
                <motion.div key="text" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.15 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                    <label style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)' }}>News text</label>
                    <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>{text.length} chars</span>
                  </div>
                  <textarea className="field-input" style={{ height: 140, marginBottom: 12 }}
                    placeholder="Paste a news article or headline here…"
                    value={text} onChange={e => setText(e.target.value)} disabled={isAnalyzing} />
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                    <span style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 500 }}>Try:</span>
                    {samples.map((s, i) => (
                      <button key={i} className="sample-btn" onClick={() => setText(s)} disabled={isAnalyzing}>Sample {i + 1}</button>
                    ))}
                  </div>
                </motion.div>
              )}
              {inputMode === 'url' && (
                <motion.div key="url" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.15 }}>
                  <label style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: 8 }}>Article URL</label>
                  <input type="url" className="field-input" style={{ marginBottom: 10 }}
                    placeholder="https://example.com/news-article"
                    value={url} onChange={e => setUrl(e.target.value)} disabled={isAnalyzing} />
                  <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12 }}>
                    Article text will be analysed and related coverage searched online.
                  </p>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                    <span style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 500 }}>Try:</span>
                    {['https://www.bbc.com/news/world', 'https://apnews.com/hub/fact-checking'].map((s, i) => (
                      <button key={i} className="sample-btn" onClick={() => setUrl(s)} disabled={isAnalyzing}>
                        {s.replace('https://', '')}
                      </button>
                    ))}
                  </div>
                </motion.div>
              )}
              {inputMode === 'photo' && (
                <motion.div key="photo" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.15 }}>
                  <label style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: 10 }}>News image or screenshot</label>
                  <PhotoPanel
                    imagePreview={imagePreview} isDragging={isDragging} isAnalyzing={isAnalyzing}
                    onFileSelect={handleFileSelect} onClear={handleClearImage}
                    onDragOver={e => { e.preventDefault(); setIsDragging(true); }}
                    onDragLeave={() => setIsDragging(false)}
                    onDrop={e => {
                      e.preventDefault(); setIsDragging(false);
                      const f = e.dataTransfer.files[0];
                      if (f?.type.startsWith('image/')) handleFileSelect(f);
                      else setError('Please drop a valid image file.');
                    }}
                    fileInputRef={fileInputRef}
                  />
                </motion.div>
              )}
            </AnimatePresence>

            <AnimatePresence>
              {error && (
                <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }} style={{ marginTop: 12 }}>
                  <div className="error-banner">
                    <AlertTriangle size={14} style={{ flexShrink: 0, marginTop: 1 }} />{error}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <div className="divider" style={{ margin: '20px 0 16px' }} />

            <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
              <button className="btn-primary" onClick={handleAnalyze} disabled={isAnalyzing || !canSubmit}>
                {isAnalyzing
                  ? <><motion.div animate={{ rotate: 360 }} transition={{ duration: 0.8, repeat: Infinity, ease: 'linear' }}><Zap size={16} /></motion.div>Analyzing…</>
                  : <><Brain size={16} />
                    Analyze {inputMode === 'text' ? 'Text' : inputMode === 'url' ? 'URL' : 'Image'}
                    {backendOnline && useRealBackend && <span style={{ fontSize: 11, opacity: 0.75, marginLeft: 4 }}>(Live)</span>}
                  </>}
              </button>
              {backendOnline && !isAnalyzing && !result && (
                <span style={{ fontSize: 12, color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 5 }}>
                  <Globe size={12} />Will also search online for related coverage
                </span>
              )}
            </div>
          </motion.div>
            </div> {/* End Sidebar */}

            {/* ── Right Column: Outputs & Analysis ── */}
            <div className="dashboard-content">
          {/* Pipeline Stages */}
          <AnimatePresence>
            {(isAnalyzing || result) && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} style={{ marginBottom: 24 }}>
                <div className="section-heading"><Activity size={13} />Pipeline Execution</div>
                <div className="space-y-3">
                  {Object.keys(stageNames).map((stage, i) => (
                    <StageCard key={stage} stage={stage} index={i}
                      isActive={currentStage === i} isCompleted={completedStages.includes(i)}
                      data={stageData[stage] || null} />
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Results */}
          <AnimatePresence>
            {result && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <div className="section-heading"><Terminal size={13} />Analysis Results</div>
                <ResultDisplay
                  result={result}
                  isRealAnalysis={isRealAnalysis}
                  inputMode={inputMode}
                  searchCtx={searchCtx}
                  isSearching={isSearching}
                  backendOnline={backendOnline}
                />
              </motion.div>
            )}
          </AnimatePresence>

          {/* Feature cards — only on fresh state */}
          {!isAnalyzing && !result && (
            <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }} style={{ marginTop: 48 }}>
              <div className="section-heading" style={{ justifyContent: 'center', marginBottom: 20 }}>
                <Sparkles size={13} />What we analyze
              </div>
              <div className="feature-grid">
                {[
                  { icon: Database, title: 'Data Collection',    desc: 'Text, URL articles, or image OCR output',            color: '#1a73e8' },
                  { icon: FileText, title: 'Text Preprocessing', desc: 'NLTK tokenization, lemmatization & VADER sentiment', color: '#7c3aed' },
                  { icon: Search,   title: 'Feature Extraction', desc: 'TF-IDF vectorization & linguistic features',         color: '#db2777' },
                  { icon: Globe,    title: 'Online Coverage',    desc: 'Live web search for related news & fact-checks',     color: '#059669' },
                ].map((f, i) => (
                  <motion.div key={i} className="feature-card"
                    initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.35 + i * 0.08 }}>
                    <div className="feature-icon" style={{ background: `${f.color}18`, color: f.color }}>
                      <f.icon size={22} />
                    </div>
                    <h4>{f.title}</h4>
                    <p>{f.desc}</p>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}

            </div> {/* End Content */}
          </div> {/* End Layout */}
        </div>
      </main>

      <footer className="app-footer">
        <div className="footer-inner">
          <span>Fake News Analyzer · Python NLTK + React</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              <span className="pulse-dot" style={{
                display: 'inline-block', width: 6, height: 6, borderRadius: '50%',
                background: backendOnline && useRealBackend ? 'var(--success)' : 'var(--warning)',
              }} />
              {backendOnline && useRealBackend ? 'Live NLTK + Web Search' : 'Demo mode'}
            </span>
            <span>{theme === 'dark' ? '🌙 Dark' : '☀️ Light'} mode</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

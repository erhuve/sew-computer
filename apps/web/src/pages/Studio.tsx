import { useEffect, useRef, useState } from "react";
import {
  Plus,
  ArrowLeft,
  Scissors,
  Layers3,
  Image,
  Save,
  Download,
  X,
  LoaderCircle,
  ArrowUpRight,
  Undo2,
  RefreshCw,
  Folder,
  FileText,
} from "lucide-react";
import {
  canonical,
  type Draft,
  type GarmentDocument,
  type ProjectState,
  type Project,
  type Job,
  type PatternGeometry,
  type ImportPreview,
} from "../../../../packages/contracts";
import Authoring from "../components/Authoring";
import PatternCanvas from "../components/PatternCanvas";
import { api, ApiError, json, downloadJson, downloadFile } from "../lib/api";
import PrivateImage from "../components/PrivateImage";
import DesignAssistant from "../components/DesignAssistant";
import { rebaseAcceptedDesign, type DesignProposal, type InterpretationStatus } from "../../../../packages/contracts/interpretation";
import "../studio.css";

const sections = [
  ["idea", "Idea"],
  ["shape", "Shape & body"],
  ["materials", "Materials"],
  ["measurements", "Measurements"],
  ["construction", "Construction"],
  ["references", "References"],
  ["review", "Review"],
];
const same = (a: unknown, b: unknown) => canonical(a) === canonical(b);
export default function Studio() {
  const [showDetails, setShowDetails] = useState(false);
  const [aiStatus,setAiStatus]=useState<InterpretationStatus|null>(null),[proposal,setProposal]=useState<DesignProposal|null>(null);
  const [authenticated, setAuthenticated] = useState<boolean | null>(null),
    [key, setKey] = useState(""),
    [projects, setProjects] = useState<Project[]>([]),
    [state, setState] = useState<ProjectState | null>(null),
    [doc, setDoc] = useState<GarmentDocument | null>(null),
    [section, setSection] = useState("idea"),
    [view, setView] = useState("design");
  const [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [notice, setNotice] = useState(""),
    [conflict, setConflict] = useState<Draft | null>(null),
    [geometry, setGeometry] = useState<PatternGeometry | null>(null),
    [selected, setSelected] = useState<string | null>(null),
    [modal, setModal] = useState<
      "create" | "revisions" | "export" | "import" | null
    >(null),
    [title, setTitle] = useState(""),
    [brief, setBrief] = useState("");
  const [revisionId, setRevisionId] = useState(""),
    [includeBody, setIncludeBody] = useState(false),
    [includeReferences, setIncludeReferences] = useState(false),
    [includePatterns, setIncludePatterns] = useState(false),
    [files, setFiles] = useState<
      { filename: string; mime: string; url: string }[]
    >([]),
    [preview, setPreview] = useState<ImportPreview | null>(null),
    [resolutions, setResolutions] = useState<
      Record<string, "keep" | "incoming">
    >({});
  const [comment, setComment] = useState(""),
    [reviewer, setReviewer] = useState(""),
    [anchor, setAnchor] = useState("/construction"),
    [undo, setUndo] = useState<GarmentDocument[]>([]);
  const modalRef = useRef<HTMLDialogElement>(null),
    restoreFocus = useRef<HTMLElement | null>(null),
    current = useRef({ state, doc }),
    saveLock = useRef(false),
    sessionEpoch = useRef(0),
    exportEpoch = useRef(0);
  current.current = { state, doc };
  const renderEpoch = sessionEpoch.current;
  const appendReference = (reference: GarmentDocument["views"][number]) => {
    if (sessionEpoch.current !== renderEpoch || current.current.state?.project.id !== state?.project.id) return;
    setDoc((latest) =>
      latest && latest.views.length < 20
        ? { ...latest, views: [...latest.views, reference] }
        : latest,
    );
  };
  const dirty = !!state && !!doc && !same(doc, state.draft.document);
  const pending = state?.jobs.find(
    (j) => j.status === "queued" || j.status === "running",
  );
  const currentJob = state?.jobs.find(job => job.revisionId === state.project.headRevisionId);
  const measurementStep = section === 'shape' && view === 'design' && !pending;
  useEffect(() => {
    document.querySelector('.editor-sidebar')?.scrollTo(0, 0);
    if (section === 'shape') document.querySelector<HTMLElement>('.body-heading')?.focus();
  }, [section, state?.project.id]);
  const fail = (e: unknown) => {
    setError(e instanceof Error ? e.message : String(e));
    if (e instanceof ApiError && e.status === 409 && e.details.latest)
      setConflict(e.details.latest as Draft);
    if (e instanceof ApiError && e.status === 401) setAuthenticated(false);
  };
  const task = async (fn: () => Promise<void>) => {
    if (busy) return;
    setBusy(true);
    setError("");
    try {
      await fn();
    } catch (e) {
      fail(e);
    } finally {
      setBusy(false);
    }
  };
  const closeModal = () => {
    if (busy) return;
    modalRef.current?.close();
    setModal(null);
    setTimeout(() => restoreFocus.current?.focus(), 0);
  };
  const openModal = (next: typeof modal) => {
    exportEpoch.current++;
    restoreFocus.current = document.activeElement as HTMLElement;
    setModal(next);
    setFiles([]);
    if (state) setRevisionId(state.project.headRevisionId || "");
  };
  const invalidateExport = () => {
    exportEpoch.current++;
    setFiles([]);
  };
  useEffect(() => {
    if (modal && !modalRef.current?.open) modalRef.current?.showModal();
  }, [modal]);
  useEffect(() => {
    api<{ authenticated: boolean }>("/auth/status")
      .then((r) => setAuthenticated(r.authenticated))
      .catch(fail);
  }, []);
  const list = async () =>
    setProjects((await api<{ projects: Project[] }>("/projects")).projects);
  useEffect(() => {
    if (authenticated) {
      list().catch(fail);
      api<InterpretationStatus>("/interpretation/status").then(setAiStatus).catch(fail);
    }
  }, [authenticated]);
  const adopt = (next: ProjectState) => {
    sessionEpoch.current++;
    setState(next);
    setDoc(structuredClone(next.draft.document));
    setConflict(null);
    setUndo([]);
    setSelected(null);
    setGeometry(null);
    setNotice("");
    setProposal(null);
    setSection(next.draft.document.garment.family === 'none' ? 'idea' : 'shape');
    setView(next.artifacts.some(artifact=>artifact.kind==='pattern-json' && artifact.revisionId === next.project.headRevisionId)?'pattern':'design');
  };
  useEffect(()=>{
    if(!state)return;
    let active=true;
    api<DesignProposal|null>(`/projects/${state.project.id}/proposals/latest`).then(value=>{if(active){setProposal(value);if(value){setSection('idea');setView('design');}}}).catch(fail);
    return ()=>{active=false;};
  },[state?.project.id]);
  const load = async (id: string) => {
    if (
      dirty &&
      !confirm(
        "Leave these unsaved edits? Download a local copy first if you want to keep them.",
      )
    )
      return;
    adopt(await api<ProjectState>(`/projects/${id}`));
  };
  useEffect(() => {
    const revision = state?.project.headRevisionId,
      project = state?.project.id;
    if (!revision || !project) {
      setGeometry(null);
      return;
    }
    let cancelled = false;
    api<PatternGeometry>(`/projects/${project}/geometry/${revision}`)
      .then((g) => {
        if (!cancelled) setGeometry(g);
      })
      .catch((e) => {
        if (!cancelled) {
          setGeometry(null);
          if (!(e instanceof ApiError && e.status === 404)) fail(e);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [
    state?.project.id,
    state?.project.headRevisionId,
    state?.artifacts.length,
  ]);
  useEffect(() => {
    if (!pending || !state) return;
    let active = true;
    const projectId = state.project.id;
    const timer = setInterval(() => {
      api<ProjectState>(`/projects/${projectId}`)
        .then((next) => {
          if (active && current.current.state?.project.id === projectId)
            setState((s) => (s ? { ...next, draft: s.draft } : next));
        })
        .catch((e) => {
          if (active) fail(e);
        });
    }, 1000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [state?.project.id, pending?.id]);
  useEffect(() => {
    const leave = (e: BeforeUnloadEvent) => {
      if (dirty) {
        e.preventDefault();
        e.returnValue = "";
      }
    };
    window.addEventListener("beforeunload", leave);
    return () => window.removeEventListener("beforeunload", leave);
  }, [dirty]);
  async function save(): Promise<Draft | null> {
    const now = current.current;
    if (!now.state || !now.doc || saveLock.current || conflict) return null;
    if (same(now.doc, now.state.draft.document)) return now.state.draft;
    const projectId = now.state.project.id,
      submitted = structuredClone(now.doc),
      epoch = sessionEpoch.current;
    saveLock.current = true;
    try {
      const draft = await api<Draft>(
        `/projects/${projectId}/draft`,
        json("PUT", {
          expectedVersion: now.state.draft.version,
          expectedRevisionId: now.state.draft.baseRevisionId,
          document: submitted,
        }),
      );
      if (
        epoch === sessionEpoch.current &&
        current.current.state?.project.id === projectId
      ) {
        setState((s) => (s ? { ...s, draft } : s));
        setNotice("Saved");
      }
      return draft;
    } finally {
      saveLock.current = false;
    }
  }
  useEffect(() => {
    if (!dirty || conflict || busy) return;
    const timer = setTimeout(() => save().catch(fail), 900);
    return () => clearTimeout(timer);
  }, [doc, state?.draft.version, conflict, busy]);
  const change = (next: GarmentDocument) => {
    if (doc) setUndo((old) => [...old.slice(-19), structuredClone(doc)]);
    setDoc(next);
    setNotice("");
  };
  async function publish() {
    const epoch = sessionEpoch.current;
    const projectId = current.current.state?.project.id;
    const draft = await save();
    if (!draft || !projectId || epoch !== sessionEpoch.current) return;
    const next = await api<ProjectState>(
      `/projects/${projectId}/revisions`,
      json("POST", {
        expectedVersion: draft.version,
        expectedRevisionId: draft.baseRevisionId,
      }),
    );
    if (epoch !== sessionEpoch.current || current.current.state?.project.id !== projectId) return;
    setState(next);
    setDoc((latest) =>
      latest && !same(latest, draft.document)
        ? latest
        : structuredClone(next.draft.document),
    );
    setConflict(null);
    setNotice("Revision saved");
    return next;
  }
  async function generate() {
    const epoch = sessionEpoch.current;
    const existing=current.current.state;
    const head=existing?.revisions.find(revision=>revision.id===existing.project.headRevisionId);
    const currentState = head&&same(current.current.doc,head.document)?existing:await publish();
    if (!currentState?.project.headRevisionId) return;
    const job = await api<Job>(
      `/projects/${currentState.project.id}/jobs`,
      json("POST", {
        revisionId: currentState.project.headRevisionId,
        requestId: crypto.randomUUID(),
      }),
    );
    if (epoch !== sessionEpoch.current || current.current.state?.project.id !== currentState.project.id) return;
    setState((s) => (s ? { ...s, jobs: [job, ...s.jobs] } : s));
    setView("pattern");
  }
  async function propose(includeReferences:boolean) {
    const epoch=sessionEpoch.current,projectId=current.current.state?.project.id;
    const draft=await save();
    if(!draft||!projectId)return;
    const result=await api<DesignProposal>(`/projects/${projectId}/proposals`,json('POST',{expectedVersion:draft.version,expectedRevisionId:draft.baseRevisionId,includeReferences,consent:true}));
    if(epoch===sessionEpoch.current&&current.current.state?.project.id===projectId)setProposal(result);
  }
  async function acceptProposal() {
    if(!proposal||!state||dirty)return;
    const epoch=sessionEpoch.current,submitted=structuredClone(current.current.doc!);
    const next=await api<ProjectState>(`/projects/${state.project.id}/proposals/${proposal.id}/accept`,json('POST',{expectedVersion:state.draft.version,expectedRevisionId:state.draft.baseRevisionId}));
    if(epoch!==sessionEpoch.current)return;
    setState(next);
    setDoc(latest=>latest?rebaseAcceptedDesign(next.draft.document,submitted,latest):structuredClone(next.draft.document));
    const hasPatternShape = next.draft.document.garment.family !== 'none';
    setProposal(null);setGeometry(null);setSection(hasPatternShape ? 'shape' : 'idea');setView('design');setNotice(hasPatternShape ? 'Design accepted. Confirm your measurements, then generate.' : 'Design notes saved. Pattern support is still needed.');
  }
  const goHome = () => {
    if (dirty && !confirm("Leave unsaved changes?")) return;
    sessionEpoch.current++;
    setState(null);
    setDoc(null);
    setConflict(null);
    setGeometry(null);
    list().catch(fail);
  };
  const references = doc?.views || [];
  return (
    <main className="studio-shell">
      <header className="studio-header">
        <button className="wordmark" onClick={goHome}>
          <Scissors size={23} />
          <span>
            sew<span className="brand-dot">.</span>
          </span>
        </button>
        <div className="header-center">
          {state ? (
            <>
              <span>{doc?.title || state.project.title}</span>
              <small>
                {conflict
                  ? "Conflict — local edits preserved"
                  : dirty
                    ? "Unsaved changes"
                    : notice || "Saved draft"}
              </small>
            </>
          ) : (
            <span>Your ideas, taking shape.</span>
          )}
        </div>
        <span className="prototype-label">DESIGN STUDIO · PRIVATE</span>
        {authenticated && (
          <button
            className="icon-button"
            aria-label="Sign out"
            onClick={() =>
              task(async () => {
                if (dirty && !confirm("Sign out with unsaved edits?")) return;
                await api("/auth/logout", json("POST", {}));
                sessionEpoch.current++;
                setAuthenticated(false);
                setState(null);
                setDoc(null);
              })
            }
          >
            <ArrowUpRight size={18} />
          </button>
        )}
      </header>
      {error && (
        <div className="alert error" role="alert">
          <span>{error}</span>
          <button aria-label="Dismiss error" onClick={() => setError("")}>
            <X size={16} />
          </button>
        </div>
      )}
      {authenticated === null ? (
        <div className="loading-state">
          <LoaderCircle className="spin" />
          Opening your studio…
        </div>
      ) : !authenticated ? (
        <section className="access-card">
          <span className="eyebrow">PRIVATE STUDIO</span>
          <h1>
            A little space
            <br />
            to make something.
          </h1>
          <p>
            Enter this studio’s owner access key. Your projects and images are
            protected independently of the preview page.
          </p>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              task(async () => {
                await api("/auth/login", json("POST", { key }));
                setKey("");
                setAuthenticated(true);
              });
            }}
          >
            <label className="field">
              <span>Owner access key</span>
              <input
                type="password"
                autoComplete="current-password"
                value={key}
                onChange={(e) => setKey(e.target.value)}
                required
              />
            </label>
            <button className="primary" disabled={busy}>
              Open studio <ArrowUpRight size={16} />
            </button>
          </form>
          <p className="fineprint">
            The key is stored privately on your Zo in this project’s local data
            directory. It is never bundled into this page.
          </p>
        </section>
      ) : !state || !doc ? (
        <section className="project-home">
          <div className="home-heading">
            <div>
              <span className="eyebrow">YOUR WORKTABLE</span>
              <h1>What will you make?</h1>
              <p>
                Start with an idea. Shape a pattern. Keep the details together.
              </p>
            </div>
            <button
              className="primary"
              onClick={() => {
                setTitle("");
                setBrief("");
                openModal("create");
              }}
            >
              <Plus size={18} />
              New garment
            </button>
          </div>
          <div className="project-grid">
            {projects.map((p) => (
              <button
                className="project-card"
                key={p.id}
                onClick={() => task(() => load(p.id))}
              >
                <div className="project-cover">
                  <Scissors size={44} strokeWidth={0.8} />
                </div>
                <div>
                  <h2>{p.title}</h2>
                  <span>
                    Updated {new Date(p.updatedAt).toLocaleDateString()}
                  </span>
                </div>
                <ArrowUpRight size={18} />
              </button>
            ))}
            <button
              className="project-card new-project"
              onClick={() => {
                setTitle("");
                setBrief("");
                openModal("create");
              }}
            >
              <Plus size={30} />
              <span>A new possibility</span>
            </button>
          </div>
          <p className="home-note">
            Describe a garment, review the design, add measurements and generate
            actual patterns with a draft tech pack. Physical fit is not simulated.
          </p>
        </section>
      ) : (
        <>
          <div className="editor-toolbar">
            <button className="text-button" onClick={goHome}>
              <ArrowLeft size={16} />
              All garments
            </button>
            <div>
              <button
                disabled={busy || !dirty || !!conflict}
                onClick={() =>
                  task(async () => {
                    await save();
                  })
                }
              >
                <Save size={15} />
                Save draft
              </button>
              <button
                className="primary"
                disabled={busy || !!conflict || !!pending || (!!doc.interpretation && doc.garment.family === 'none')}
                onClick={() => task(generate)}
              >
                <Layers3 size={15} />
                {pending ? "Generating…" : "Save & generate"}
              </button>
              <button
                onClick={() => openModal("revisions")}
              >
                <Download size={15} />
                Revisions & export
              </button>
            </div>
          </div>
          {conflict && (
            <div className="alert conflict" role="alert">
              <div>
                <strong>Another tab changed this draft.</strong>
                <p>
                  Your local edits are still here. Download them before loading
                  the newer saved version.
                </p>
              </div>
              <button
                onClick={() => downloadJson("local-unsaved-draft.json", doc)}
              >
                Download local edits
              </button>
              <button
                onClick={() =>
                  task(async () => {
                    adopt(
                      await api<ProjectState>(`/projects/${state.project.id}`),
                    );
                  })
                }
              >
                Load saved version
              </button>
            </div>
          )}
          <div className={`worktable${measurementStep ? ' measurement-step' : ''}`}>
            <aside className="editor-sidebar">
              <nav className="section-nav" aria-label="Garment details">
                {sections.filter(([id]) => !measurementStep || showDetails || ['idea','shape','references'].includes(id!)).map(([id, label]) => (
                  <button
                    key={id}
                    aria-current={section === id ? "page" : undefined}
                    onClick={() => { setSection(id!); if(id === 'shape' || id === 'idea') setView('design'); }}
                  >
                    {label}
                  </button>
                ))}
                {measurementStep && <button aria-expanded={showDetails} onClick={() => setShowDetails(!showDetails)}>{showDetails ? 'Fewer details' : 'More details'}</button>}
              </nav>
              <Authoring
                key={state.project.id + section}
                doc={doc}
                onChange={change}
                onReference={appendReference}
                projectId={state.project.id}
                section={section}
                onError={setError}
                comments={state.comments}
                anchor={selected ? `panel:${selected}` : null}
              />
              <div className="editor-footer">
                <button
                  disabled={!undo.length}
                  onClick={() => {
                    const prev = undo.at(-1);
                    if (prev) {
                      setDoc(prev);
                      setUndo((v) => v.slice(0, -1));
                    }
                  }}
                >
                  <Undo2 size={15} />
                  Undo edit
                </button>
                <small>Private · revisioned</small>
              </div>
            </aside>
            <section className="visual-workspace">
              <div
                className="view-tabs"
                role="tablist"
                aria-label="Visual view"
              >
                <button role="tab" aria-selected={view==='design'} onClick={()=>setView('design')}><FileText size={16}/>Design</button>
                <button
                  role="tab"
                  aria-selected={view === "pattern"}
                  onClick={() => setView("pattern")}
                >
                  <Layers3 size={16} />
                  Pattern
                </button>
                <button
                  role="tab"
                  aria-selected={view === "references"}
                  onClick={() => setView("references")}
                >
                  <Image size={16} />
                  Idea & references
                </button>
              </div>
              {pending && (
                <div className="job-progress" role="status">
                  <LoaderCircle className="spin" size={16} />
                  {pending.status === "queued"
                    ? "Waiting for the engine"
                    : "Generating actual panels"}
                  <button
                    onClick={() =>
                      task(async () => {
                        const job = await api<Job>(
                          `/projects/${state.project.id}/jobs/${pending.id}/cancel`,
                          json("POST", {}),
                        );
                        setState((s) =>
                          s
                            ? {
                                ...s,
                                jobs: s.jobs.map((j) =>
                                  j.id === job.id ? job : j,
                                ),
                              }
                            : s,
                        );
                      })
                    }
                  >
                    Cancel
                  </button>
                </div>
              )}
              {currentJob?.status === "failed" && (
                <div className="alert error">
                  Pattern generation failed: {currentJob.error}. Your draft
                  and saved revision are intact.
                </div>
              )}
              {geometry &&
                (dirty ||
                  !same(
                    doc,
                    state.revisions.find(
                      (r) => r.id === state.project.headRevisionId,
                    )?.document,
                  )) && (
                  <div className="geometry-stale">
                    These panels belong to saved revision{" "}
                    {
                      state.revisions.find(
                        (r) => r.id === state.project.headRevisionId,
                      )?.number
                    }
                    . Save & generate to update them.
                  </div>
                )}
              {view === 'design' ? <DesignAssistant key={state.project.id} doc={doc} status={aiStatus} proposal={proposal} busy={busy||!!conflict}
                stale={!!proposal&&(dirty||proposal.baseVersion!==state.draft.version||proposal.baseRevisionId!==state.draft.baseRevisionId)}
                onPropose={images=>task(()=>propose(images))} onAccept={()=>task(acceptProposal)} onMeasurements={()=>setSection('shape')}/> : view === "pattern" ? (
                <PatternCanvas
                  geometry={geometry}
                  selected={selected}
                  onSelect={setSelected}
                />
              ) : (
                <div className="reference-board">
                  {references.length ? (
                    references.map((r) => (
                      <figure key={r.id}>
                        <PrivateImage
                          src={`/api/projects/${state.project.id}/references/${r.assetId}`}
                          alt={r.caption || `${r.role} ${r.kind}`}
                        />
                        <figcaption>
                          <span>
                            {r.role} · {r.kind}
                          </span>
                          {r.caption}
                        </figcaption>
                      </figure>
                    ))
                  ) : (
                    <div className="canvas-empty">
                      <Image size={38} strokeWidth={1} />
                      <h2>Your idea, in view.</h2>
                      <p>
                        Add photos, sketches or your own technical flats. They
                        won’t be presented as a simulation.
                      </p>
                      <button onClick={() => setSection("references")}>
                        Add a reference
                      </button>
                    </div>
                  )}
                </div>
              )}
              {geometry && (
                <details className="geometry-notes">
                  <summary>
                    Engine scope & open questions · {geometry.warnings.length}
                  </summary>
                  <ul>
                    {geometry.warnings.map((warning, i) => (
                      <li key={i}>{warning}</li>
                    ))}
                  </ul>
                </details>
              )}
              {selected && geometry && (
                <div className="selected-panel">
                  <span>
                    {geometry.panels.find((p) => p.id === selected)?.name} ·{" "}
                    {
                      geometry.stitches.filter(
                        (s) => s.panelA === selected || s.panelB === selected,
                      ).length
                    }{" "}
                    seam links
                  </span>
                  <button
                    onClick={() => {
                      setAnchor(`panel:${selected}`);
                      setSection("construction");
                    }}
                  >
                    Add a callout
                  </button>
                </div>
              )}
            </section>
          </div>
        </>
      )}
      <dialog
        ref={modalRef}
        className="studio-dialog"
        onCancel={(e) => {
          e.preventDefault();
          closeModal();
        }}
        onClose={() => setModal(null)}
      >
        <button
          className="dialog-close icon-button"
          aria-label="Close dialog"
          onClick={closeModal}
        >
          <X size={20} />
        </button>
        {modal === "create" && (
          <form
            onSubmit={(e) => {
              e.preventDefault();
              task(async () => {
                const next = await api<ProjectState>(
                  "/projects",
                  json("POST", { title: title.trim(), brief }),
                );
                adopt(next);
                setSection("idea");
                closeModal();
              });
            }}
          >
            <span className="eyebrow">START SOMETHING</span>
            <h2>A garment of your own.</h2>
            <label className="field">
              <span>Garment name</span>
              <input
                autoFocus
                maxLength={160}
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                required
                placeholder="The everyday overshirt"
              />
            </label>
            <label className="field">
              <span>Your idea</span>
              <textarea
                maxLength={8000}
                value={brief}
                onChange={(e) => setBrief(e.target.value)}
                placeholder="Shape, feeling, details. Nothing needs to be decided yet."
              />
            </label>
            <p className="fineprint">
              Next, interpret your idea into an editable design. You review
              suggestions before they become a saved revision.
            </p>
            <button className="primary" disabled={busy || !title.trim()}>
              Create garment <ArrowUpRight size={16} />
            </button>
          </form>
        )}
        {modal === "revisions" && state && (
          <>
            <span className="eyebrow">THE CHANGE RECORD</span>
            <h2>Save the moment.</h2>
            <p>
              Exports always use an immutable revision, never your changing
              draft.
            </p>
            <button
              className="primary"
              disabled={busy || !!conflict}
              onClick={() =>
                task(async () => {
                  await publish();
                })
              }
            >
              <Save size={16} />
              Save new revision
            </button>
            <div className="revision-list">
              {state.revisions.map((r) => (
                <article key={r.id}>
                  <div>
                    <strong>Revision {r.number}</strong>
                    <span>
                      {r.document.title} · {r.id.slice(0, 8)}
                    </span>
                  </div>
                  <button
                    onClick={() => {
                      setRevisionId(r.id);
                      setFiles([]);
                      setModal("export");
                    }}
                  >
                    Export draft
                  </button>
                </article>
              ))}
            </div>
            {!state.revisions.length && (
              <p>
                No saved revisions yet. Your editable draft is retained
                separately.
              </p>
            )}
            <h3>Record a review note</h3>
            <label className="field">
              <span>Revision</span>
              <select
                value={revisionId}
                onChange={(e) => setRevisionId(e.target.value)}
              >
                <option value="">Choose a revision</option>
                {state.revisions.map((r) => (
                  <option key={r.id} value={r.id}>
                    Revision {r.number}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Reported reviewer (optional)</span>
              <input
                value={reviewer}
                onChange={(e) => setReviewer(e.target.value)}
              />
            </label>
            <label className="field">
              <span>Field / panel anchor</span>
              <input
                value={anchor}
                onChange={(e) => setAnchor(e.target.value)}
              />
            </label>
            <label className="field">
              <span>Feedback</span>
              <textarea
                value={comment}
                onChange={(e) => setComment(e.target.value)}
              />
            </label>
            <button
              disabled={busy || !revisionId || !comment.trim()}
              onClick={() =>
                task(async () => {
                  await api(
                    `/projects/${state.project.id}/comments`,
                    json("POST", {
                      revisionId,
                      anchor,
                      text: comment,
                      reportedReviewer: reviewer,
                    }),
                  );
                  setState((s) => (s ? { ...s, comments: [] } : s));
                  const fresh = await api<ProjectState>(
                    `/projects/${state.project.id}`,
                  );
                  setState((s) => (s ? { ...s, comments: fresh.comments } : s));
                  setComment("");
                  setNotice("Review note recorded by owner");
                })
              }
            >
              Record note
            </button>
            <hr />
            <label className="upload-button">
              <FileText size={16} />
              Import edited manifest
              <input
                type="file"
                accept="application/json,.json"
                onChange={async (e) => {
                  const file = e.target.files?.[0];
                  if (!file) return;
                  await task(async () => {
                    if (dirty)
                      throw new Error(
                        "Save your current edits before previewing an import.",
                      );
                    if (file.size > 2 * 1024 * 1024)
                      throw new Error("Manifest must be under 2 MB.");
                    const p = await api<ImportPreview>(
                      `/projects/${state.project.id}/imports/preview`,
                      json("POST", { manifest: JSON.parse(await file.text()) }),
                    );
                    setPreview(p);
                    setResolutions({});
                    setModal("import");
                  });
                  e.target.value = "";
                }}
              />
            </label>
            <button
              className="danger-link"
              onClick={() =>
                task(async () => {
                  if (
                    !confirm(
                      "Permanently delete this garment, revisions and private files?",
                    )
                  )
                    return;
                  await api(`/projects/${state.project.id}`, {
                    method: "DELETE",
                  });
                  closeModal();
                  sessionEpoch.current++;
                  setState(null);
                  setDoc(null);
                  await list();
                })
              }
            >
              Delete this garment
            </button>
          </>
        )}
        {modal === "export" && state && (
          <>
            <span className="eyebrow">A DOCUMENT TO DISCUSS</span>
            <h2>Export a draft tech pack.</h2>
            <p>
              Revision{" "}
              {state.revisions.find((r) => r.id === revisionId)?.number} ·{" "}
              {revisionId.slice(0, 8)}. Not manufacturing approval.
            </p>
            <label className="check-field">
              <input
                type="checkbox"
                checked={includeBody}
                onChange={(e) => { invalidateExport(); setIncludeBody(e.target.checked); }}
              />
              Include private body inputs
            </label>
            <label className="check-field">
              <input
                type="checkbox"
                checked={includeReferences}
                onChange={(e) => { invalidateExport(); setIncludeReferences(e.target.checked); }}
              />
              Include private photos, sketches and technical views
            </label>
            <label className="check-field">
              <input
                type="checkbox"
                checked={includePatterns}
                onChange={(e) => { invalidateExport(); setIncludePatterns(e.target.checked); }}
              />
              Include generated pattern references
            </label>
            <p className="fineprint">
              Geometry and free-text details can reveal body information even
              when the body-input fields are omitted. Review the files before
              sharing. Patterns are not cutting-ready.
            </p>
            <button
              className="primary"
              disabled={busy}
              onClick={() =>
                task(async () => {
                  const requestedExportEpoch = exportEpoch.current;
                  const requestedSessionEpoch = sessionEpoch.current;
                  const result = await api<{
                    snapshotId: string;
                    files: { filename: string; mime: string; url: string }[];
                  }>(
                    `/projects/${state.project.id}/exports`,
                    json("POST", {
                      revisionId,
                      disclosure: {
                        includeBody,
                        includeReferences,
                        includePatterns,
                      },
                    }),
                  );
                  if (requestedExportEpoch === exportEpoch.current && requestedSessionEpoch === sessionEpoch.current)
                    setFiles(result.files);
                })
              }
            >
              {busy ? (
                <LoaderCircle className="spin" size={16} />
              ) : (
                <Download size={16} />
              )}
              Build review package
            </button>
            <div className="download-list">
              {files.map((f) => (
                <a
                  key={f.filename}
                  href={f.url.startsWith("/api/") ? f.url : "/api" + f.url}
                  download={f.filename}
                  onClick={(event) => {
                    event.preventDefault();
                    task(() => downloadFile(f.url, f.filename));
                  }}
                >
                  <FileText size={16} />
                  {f.filename}
                  <ArrowUpRight size={14} />
                </a>
              ))}
            </div>
          </>
        )}
        {modal === "import" && state && preview && (
          <>
            <span className="eyebrow">REVIEW BEFORE APPLYING</span>
            <h2>Bring feedback back.</h2>
            <p>
              Only disclosed, editable fields can change. Private omitted fields
              remain untouched.
            </p>
            {preview.warnings.map((w, i) => (
              <p key={i}>{w}</p>
            ))}
            {preview.changes.map((c) => (
              <div className="import-change" key={c.path}>
                <code>{c.path}</code>
                {c.conflict && <>
                  <p>Original exported value</p>
                  <pre>{JSON.stringify(c.baseline, null, 2) ?? "Not present"}</pre>
                  <p>Current draft value</p>
                  <pre>{JSON.stringify(c.before, null, 2) ?? "Not present"}</pre>
                </>}
                <p>Incoming value</p>
                <pre>{JSON.stringify(c.after, null, 2)}</pre>
                {c.conflict ? (
                  <label>
                    Conflict decision
                    <select
                      value={resolutions[c.path] || ""}
                      onChange={(e) =>
                        setResolutions({
                          ...resolutions,
                          [c.path]: e.target.value as "keep" | "incoming",
                        })
                      }
                    >
                      <option value="">Choose…</option>
                      <option value="keep">Keep current value</option>
                      <option value="incoming">Use incoming value</option>
                    </select>
                  </label>
                ) : (
                  <small>No conflicting local edit</small>
                )}
              </div>
            ))}
            {!preview.changes.length && <p>No editable changes were found.</p>}
            <button
              className="primary"
              disabled={
                busy ||
                !preview.changes.length ||
                preview.changes.some((c) => c.conflict && !resolutions[c.path])
              }
              onClick={() =>
                task(async () => {
                  const next = await api<ProjectState>(
                    `/projects/${state.project.id}/imports/${preview.id}/accept`,
                    json("POST", {
                      expectedVersion: preview.baseVersion,
                      expectedRevisionId: preview.baseRevisionId,
                      resolutions,
                    }),
                  );
                  adopt(next);
                  closeModal();
                  setNotice(
                    "Feedback applied to draft; save a new revision when ready",
                  );
                })
              }
            >
              Apply reviewed changes
            </button>
          </>
        )}
      </dialog>
    </main>
  );
}

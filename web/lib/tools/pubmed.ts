// PubMed E-utilities client. No API key required for ≤3 req/s; we add `tool=`
// and `email=` params per NCBI etiquette guidelines.
//
// Two-step flow: esearch.fcgi → list of PMIDs, efetch.fcgi → abstracts XML.
// We parse the XML with a permissive regex pass so we don't drag in an XML
// dependency for ~5-10 records per query.

const NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils";
const NCBI_TOOL = "medomni";
const NCBI_EMAIL = "b@thegoatnote.com";
const TIMEOUT_MS = 8000;
const MAX_RESULTS_HARD_CAP = 10;

export interface PubMedRecord {
  pmid: string;
  title: string;
  journal: string;
  year: string;
  authors: string;
  abstract: string;
  url: string;
}

export interface PubMedSearchResult {
  query: string;
  count: number;
  records: PubMedRecord[];
}

function withTimeout<T>(p: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
    p.then(
      (v) => {
        clearTimeout(t);
        resolve(v);
      },
      (e) => {
        clearTimeout(t);
        reject(e);
      },
    );
  });
}

function decodeEntities(s: string): string {
  return s
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&amp;/g, "&")
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&#x([0-9a-fA-F]+);/g, (_, h) => String.fromCodePoint(parseInt(h, 16)))
    .replace(/&#(\d+);/g, (_, n) => String.fromCodePoint(parseInt(n, 10)));
}

function stripTags(s: string): string {
  return decodeEntities(s.replace(/<[^>]+>/g, "")).replace(/\s+/g, " ").trim();
}

function pickFirst(xml: string, tag: string): string {
  const m = xml.match(new RegExp(`<${tag}[^>]*>([\\s\\S]*?)<\\/${tag}>`));
  return m ? stripTags(m[1]) : "";
}

function parseArticle(xml: string): PubMedRecord {
  const pmid = pickFirst(xml, "PMID");
  const title = pickFirst(xml, "ArticleTitle");
  const journal = pickFirst(xml, "Title") || pickFirst(xml, "ISOAbbreviation");
  const year = pickFirst(xml, "Year");

  const authorMatches = [...xml.matchAll(/<Author[\s\S]*?<\/Author>/g)].slice(0, 3);
  const authorNames = authorMatches.map((m) => {
    const last = pickFirst(m[0], "LastName");
    const init = pickFirst(m[0], "Initials");
    return last && init ? `${last} ${init}` : last;
  }).filter(Boolean);
  const authors = authorNames.length > 0
    ? authorNames.join(", ") + (authorMatches.length === 3 ? " et al." : "")
    : "";

  const abstractParts = [...xml.matchAll(/<AbstractText[^>]*>([\s\S]*?)<\/AbstractText>/g)];
  const abstract = abstractParts.map((m) => stripTags(m[1])).join("\n").trim();

  return {
    pmid,
    title,
    journal,
    year,
    authors,
    abstract,
    url: pmid ? `https://pubmed.ncbi.nlm.nih.gov/${pmid}/` : "",
  };
}

export async function pubmedSearch(args: {
  query: string;
  maxResults?: number;
}): Promise<PubMedSearchResult> {
  const query = args.query.trim();
  if (!query) {
    return { query, count: 0, records: [] };
  }
  const maxResults = Math.min(Math.max(args.maxResults ?? 5, 1), MAX_RESULTS_HARD_CAP);

  const esearchUrl = new URL(`${NCBI_BASE}/esearch.fcgi`);
  esearchUrl.searchParams.set("db", "pubmed");
  esearchUrl.searchParams.set("term", query);
  esearchUrl.searchParams.set("retmax", String(maxResults));
  esearchUrl.searchParams.set("retmode", "json");
  esearchUrl.searchParams.set("sort", "relevance");
  esearchUrl.searchParams.set("tool", NCBI_TOOL);
  esearchUrl.searchParams.set("email", NCBI_EMAIL);

  const esearchRes = await withTimeout(fetch(esearchUrl.toString()), TIMEOUT_MS, "pubmed esearch");
  if (!esearchRes.ok) {
    throw new Error(`pubmed esearch ${esearchRes.status}`);
  }
  const esearchJson = (await esearchRes.json()) as {
    esearchresult?: { idlist?: string[]; count?: string };
  };
  const ids = esearchJson.esearchresult?.idlist ?? [];
  const totalCount = parseInt(esearchJson.esearchresult?.count ?? "0", 10) || 0;

  if (ids.length === 0) {
    return { query, count: totalCount, records: [] };
  }

  const efetchUrl = new URL(`${NCBI_BASE}/efetch.fcgi`);
  efetchUrl.searchParams.set("db", "pubmed");
  efetchUrl.searchParams.set("id", ids.join(","));
  efetchUrl.searchParams.set("retmode", "xml");
  efetchUrl.searchParams.set("rettype", "abstract");
  efetchUrl.searchParams.set("tool", NCBI_TOOL);
  efetchUrl.searchParams.set("email", NCBI_EMAIL);

  const efetchRes = await withTimeout(fetch(efetchUrl.toString()), TIMEOUT_MS, "pubmed efetch");
  if (!efetchRes.ok) {
    throw new Error(`pubmed efetch ${efetchRes.status}`);
  }
  const xml = await efetchRes.text();

  const articleBlocks = [...xml.matchAll(/<PubmedArticle[\s\S]*?<\/PubmedArticle>/g)].map((m) => m[0]);
  const records = articleBlocks.map(parseArticle).filter((r) => r.pmid);

  return { query, count: totalCount, records };
}

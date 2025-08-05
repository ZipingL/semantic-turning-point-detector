import { Message } from "./Message";
import { TurningPointDetectorConfig } from "./types";

/**
 * Defines the formatting style for replaced headings.
 * - 'plain': Just the heading text (removes '#' markers only).
 * - 'bold': Surrounds the heading text with '**'.
 * - 'italic': Surrounds the heading text with '*'.
 * - 'bold-italic': Surrounds the heading text with '***'.
 * - 'prefix': Prepends a specific string (defined in `headingPrefix`) to the heading text.
 */
export type HeadingStyle = 'plain' | 'bold' | 'italic' | 'bold-italic' | 'prefix';

/**
 * Configuration options for selectivelyStripMarkdown function.
 */
export type StripMarkdownOptions = {
  /**
   * If true, removes list markers (*, -, +, 1.) while keeping the item text.
   * @default false
   */
  removeLists?: boolean;

  /**
   * Defines how heading syntax (#) should be replaced.
   * @default 'bold'
   */
  headingStyle?: HeadingStyle;

  /**
   * The prefix string to use when `headingStyle` is 'prefix'.
   * @default 'heading: '
   */
  headingPrefix?: string;
}

/**
 * Selectively removes or reformats Markdown elements like headings and optionally lists.
 * Headings (#) are replaced based on the specified `headingStyle`.
 * Lists (*, -, +, 1.) can optionally be stripped to plain text (controlled by `removeLists`).
 * Content remains on the same line, and overall newlines are preserved.
 *
 * @param markdown The input Markdown string.
 * @param options Configuration options for stripping and formatting.
 * @returns The processed string.
 */
export function selectivelyStripMarkdown(
  markdown: string,
  opt: StripMarkdownOptions = {}
): string {
  const {
    removeLists   = false,
    headingStyle  = "bold",
    headingPrefix = "heading: "
  } = opt;

  // 1. headings
  const fmt = (txt: string) => {
    switch (headingStyle) {
      case "italic":       return `*${txt}*`;
      case "bold-italic":  return `***${txt}***`;
      case "prefix":       return `${headingPrefix}${txt}`;
      case "plain":        return txt;
      default:             return `**${txt}**`;
    }
  };
  let out = markdown.replace(/^#{1,6}\s+(.*)$/gm, (_, h) => fmt(h.trim()));

  // 2. lists
  if (removeLists) {
    out = out
      .replace(/^(\s*)[-*+]\s+/gm, "$1")   // unordered
      .replace(/^(\s*)\d+\.\s+/gm, "$1");  // ordered
  }
  return out;
}


export type OutputStyle = "modular" | "markdown";

function htmlEscape(s: string) {
  return s.replace(/[&<>"']/g, c =>
    ({ "&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;" }[c]!)
  );
}

export function returnFormattedMessageContent(
  cfg:  Partial<TurningPointDetectorConfig>,
  msg:  Message,
  dim            = 0,

  opt: {
    outputStyle?:  OutputStyle;
    markdownOpts?: StripMarkdownOptions;
    maxLen?:       number;
    addHeader?:    boolean;
  } = {}
): string {

  // ---------------- config ----------------
  const style   = opt.outputStyle ?? "markdown";
  const maxLen  = Math.min(
    opt.maxLen ?? cfg.max_character_length ?? 20_000,
    dim === 0 ? 8_000 : 20_000
  );

  // ---------------- body ----------------
  const plain = selectivelyStripMarkdown(msg.message, opt.markdownOpts);
  const body  = plain.slice(0, maxLen);
  const truncated =
    plain.length > maxLen ? `\n[truncated, original ${plain.length} chars]` : "";

  const indented = body.split("\n").map(l => "    " + l).join("\n");

  // ---------------- meta ----------------
  const kind  = dim === 0 ? "message" : "turning_point";
  const head  = opt?.addHeader ? `${msg.author}\n` : "";
  const attrs = `${
    dim === 0 ? 'author' : 'source'
  }="${htmlEscape(msg.author)}" id="${msg.id}" dim="${dim}"`;

  // ---------------- output modes ----------------
  if (style === "markdown") {
    return `${head}**${kind.toUpperCase()} (${msg.id})**\n\n\`\`\`text\n${indented}${truncated}\n\`\`\`\n`;
  }

  // modular (XML-ish)
  return `${head}<${kind} ${attrs}>\n${indented}${truncated}\n</${kind}>\n`;
}

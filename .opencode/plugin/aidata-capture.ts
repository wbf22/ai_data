import type { Plugin } from "@opencode-ai/plugin"
import { mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { homedir } from "node:os"
import { join } from "node:path"

const DEFAULT_DIR = join(homedir(), ".local", "share", "opencode", "aidata-capture")
const MAX_PROMPTS_PER_SESSION = 25

function resolveDir(options: Record<string, unknown> | undefined): string {
  if (typeof options?.dir === "string") return options.dir
  if (process.env.AIDATA_CAPTURE_DIR) return process.env.AIDATA_CAPTURE_DIR
  return DEFAULT_DIR
}

// Heuristic tag so consumers can tell the real agent prompt apart from the
// synthetic title/summary requests, which also flow through this hook.
function classifySystem(system: string[]): string {
  const head = system.join("\n").slice(0, 600)
  if (/title generator/i.test(head)) return "title"
  if (/interactive CLI tool|You are opencode/i.test(head)) return "primary"
  return "other"
}

// Persists the rendered system prompt and the tool definitions that opencode
// sends to the model. Neither appears in `opencode export`, so this is the only
// way to reconstruct the exact model input for a session.
const AidataCapture: Plugin = async (input, options) => {
  const dir = resolveDir(options)
  try {
    mkdirSync(dir, { recursive: true })
  } catch {
    // best effort: never break a session because capture failed
  }

  // Effect schemas (what the hook exposes) are converted to real JSON Schema
  // when effect is resolvable; otherwise the raw schema is stored.
  let toJsonSchema: ((schema: unknown) => { schema: unknown }) | undefined
  try {
    const mod = await import("effect/Schema")
    if (typeof mod.toJsonSchemaDocument === "function") {
      toJsonSchema = mod.toJsonSchemaDocument as typeof toJsonSchema
    }
  } catch {
    toJsonSchema = undefined
  }

  const normalizeParameters = (parameters: unknown): unknown => {
    if (!toJsonSchema) return parameters
    try {
      return toJsonSchema(parameters).schema
    } catch {
      return parameters
    }
  }

  const systemSeen = new Map<string, Set<string>>()
  const toolDefs = new Map<string, { description: string; parameters: unknown }>()
  let toolDefsCache = ""
  let lastWrittenSession: string | undefined
  // tool.definition gives no sessionID, so we remember the most recent session
  // seen by a session-aware hook and snapshot tool defs against it.
  let currentSession: string | undefined

  const systemPath = (sessionID: string) => join(dir, `system-${sessionID}.json`)
  const toolsPath = (sessionID: string) => join(dir, `tools-${sessionID}.json`)

  const persistToolDefs = (sessionID?: string): void => {
    if (!sessionID || toolDefs.size === 0) return
    const serialized = JSON.stringify(Object.fromEntries(toolDefs), null, 2)
    if (serialized === toolDefsCache && sessionID === lastWrittenSession) return
    toolDefsCache = serialized
    lastWrittenSession = sessionID
    writeFileSync(toolsPath(sessionID), serialized)
  }

  return {
    // The only hook that reliably carries a sessionID before the model call, so
    // it both marks the active session and snapshots the current tool defs.
    "chat.params": async (hookInput) => {
      try {
        currentSession = hookInput.sessionID
        persistToolDefs(currentSession)
      } catch {
        // ignore
      }
    },

    "experimental.chat.system.transform": async (hookInput, output) => {
      try {
        const sessionID = hookInput.sessionID ?? "unknown"
        currentSession = hookInput.sessionID
        const system = Array.isArray(output.system) ? output.system : []
        if (system.length === 0) return

        const key = system.join("\u0000")
        let seen = systemSeen.get(sessionID)
        if (!seen) {
          seen = new Set()
          systemSeen.set(sessionID, seen)
        }
        if (seen.has(key)) return
        seen.add(key)

        const file = systemPath(sessionID)
        let record: { sessionID: string; worktree: string; prompts: unknown[] } = {
          sessionID,
          worktree: input.worktree,
          prompts: [],
        }
        try {
          record = JSON.parse(readFileSync(file, "utf8"))
        } catch {
          // first time seeing this session
        }
        record.sessionID = sessionID
        record.worktree = input.worktree
        record.prompts.push({
          kind: classifySystem(system),
          model: hookInput.model,
          capturedAt: new Date().toISOString(),
          system,
        })
        if (record.prompts.length > MAX_PROMPTS_PER_SESSION) {
          record.prompts = record.prompts.slice(-MAX_PROMPTS_PER_SESSION)
        }
        writeFileSync(file, JSON.stringify(record, null, 2))
        if (hookInput.sessionID) persistToolDefs(sessionID)
      } catch {
        // ignore
      }
    },

    "tool.definition": async (hookInput, output) => {
      try {
        toolDefs.set(hookInput.toolID, {
          description: output.description,
          parameters: normalizeParameters(output.parameters),
        })
        persistToolDefs(currentSession)
      } catch {
        // ignore
      }
    },
  }
}

export default AidataCapture

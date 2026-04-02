import { appendFile, mkdir } from 'fs/promises'
import { homedir } from 'os'
import { dirname, isAbsolute, join, sep } from 'path'
import { getClaudeConfigHomeDir, isEnvTruthy } from '../../utils/envUtils.js'
import { logForDebugging } from '../../utils/debug.js'
import { jsonStringify } from '../../utils/slowOperations.js'

/**
 * When CLAUDE_CODE_LOG_API_MESSAGES=1, append each API request's `messages`
 * array (as sent to the model, after cache breakpoints) as one JSON line to a log file.
 *
 * - Default file: ~/.claude/claude.info
 * - Override path: CLAUDE_CODE_LOG_API_MESSAGES_PATH
 *   - Supports `~/...` (shell-style tilde is expanded; Node does not do this by default)
 *   - If the path ends with `/` or `\`, it is treated as a directory and
 *     messages are appended to `claude.info` inside that directory.
 *   - Otherwise: absolute path as-is, or relative to process.cwd()
 *
 * Security: may contain secrets and PII — do not enable in shared environments.
 */

let appendChain: Promise<void> = Promise.resolve()

export function isApiMessagesFileLoggingEnabled(): boolean {
  return isEnvTruthy(process.env.CLAUDE_CODE_LOG_API_MESSAGES)
}

function expandUserPath(p: string): string {
  if (p === '~') return homedir()
  if (p.startsWith(`~${sep}`) || p.startsWith('~/')) {
    return join(homedir(), p.slice(2))
  }
  return p
}

function resolveLogFilePath(): string {
  const raw = process.env.CLAUDE_CODE_LOG_API_MESSAGES_PATH?.trim()
  if (!raw) {
    return join(getClaudeConfigHomeDir(), 'claude.info')
  }
  let p = expandUserPath(raw)
  const isDirHint =
    p.endsWith('/') || p.endsWith('\\') || p.endsWith(sep)
  if (isDirHint) {
    p = p.replace(/[/\\]+$/, '')
    return join(p, 'claude.info')
  }
  if (isAbsolute(p)) {
    return p
  }
  return join(process.cwd(), p)
}

export function logApiMessagesToFile(payload: {
  querySource?: string
  model: string
  messages: unknown[]
}): void {
  if (!isApiMessagesFileLoggingEnabled()) return

  const filePath = resolveLogFilePath()
  const record = {
    ts: new Date().toISOString(),
    querySource: payload.querySource,
    model: payload.model,
    messageCount: payload.messages.length,
    messages: payload.messages,
  }
  const line = `${jsonStringify(record)}\n`

  appendChain = appendChain
    .then(async () => {
      await mkdir(dirname(filePath), { recursive: true })
      await appendFile(filePath, line, 'utf8')
    })
    .catch((e: unknown) => {
      logForDebugging(
        `[log-api-messages] write failed (${filePath}): ${e instanceof Error ? e.message : String(e)}`,
        { level: 'warn' },
      )
    })
}

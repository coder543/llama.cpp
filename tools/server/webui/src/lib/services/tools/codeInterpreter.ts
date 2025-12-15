import type { ApiToolDefinition } from '$lib/types';
import { registerTool } from './registry';

export const CODE_INTERPRETER_JS_TOOL_NAME = 'code_interpreter_javascript';

export const codeInterpreterToolDefinition: ApiToolDefinition = {
	type: 'function',
	function: {
		name: CODE_INTERPRETER_JS_TOOL_NAME,
		description:
			'Execute JavaScript in a sandboxed environment. Returns console output and the final evaluated value.',
		parameters: {
			type: 'object',
			properties: {
				code: {
					type: 'string',
					description: 'JavaScript source code to run.'
				}
			},
			required: ['code']
		}
	}
};

export interface CodeInterpreterResult {
	result?: string;
	logs: string[];
	error?: string;
	errorLine?: number;
	errorLineContent?: string;
	errorStack?: string;
	errorFrame?: string;
}

export async function runCodeInterpreter(
	code: string,
	timeoutMs = 5000
): Promise<CodeInterpreterResult> {
	return new Promise((resolve) => {
		const logs: string[] = [];

		const workerSource = `
      const send = (msg) => postMessage(msg);

      const logs = [];
      ['log','info','warn','error'].forEach((level) => {
        const orig = console[level];
        console[level] = (...args) => {
          const text = args.map((a) => {
            try { return typeof a === 'string' ? a : JSON.stringify(a); }
            catch { return String(a); }
          }).join(' ');
          logs.push(text);
          send({ type: 'log', level, text });
          if (orig) try { orig.apply(console, args); } catch { /* ignore */ }
        };
      });

      const transformCode = (code) => {
        const lines = (code ?? '').split('\\n');
        let i = lines.length - 1;
        while (i >= 0 && lines[i].trim() === '') i--;
        if (i < 0) return code ?? '';

        const last = lines[i];
        const trimmed = last.trim();

        // If already returns, leave as-is.
        if (/^return\\b/.test(trimmed)) return code ?? '';

        // If the last line starts/ends with block delimiters, keep code as-is (likely a statement block).
        if (/^[}\\])]/.test(trimmed) || trimmed.endsWith('{') || trimmed.endsWith('};')) {
          return code ?? '';
        }

        // If it's a declaration, return that identifier.
        const declMatch = trimmed.match(/^(const|let|var)\\s+([A-Za-z_$][\\w$]*)/);
        if (declMatch) {
          const name = declMatch[2];
          lines.push(\`return \${name};\`);
          return lines.join('\\n');
        }

        // Default: treat last statement as expression and return it.
        lines[i] = \`return (\${trimmed.replace(/;$/, '')});\`;
        return lines.join('\\n');
      };

      const run = async (code) => {
        try {
          const executable = transformCode(code);
          const markerStart = '__USER_CODE_START__';
          const markerEnd = '__USER_CODE_END__';
          const USER_OFFSET = 2; // lines before user code in wrapped string
          const wrapped = \`"use strict";\\n// \${markerStart}\\n\${executable}\\n// \${markerEnd}\`;
          const fn = new Function(wrapped);
          const result = await fn();
          send({ type: 'done', result, logs });
        } catch (err) {
          let lineNum = undefined;
          let lineText = undefined;
          try {
            const stack = String(err?.stack ?? '');
            const match =
              stack.match(/<anonymous>:(\\d+):(\\d+)/) ||
              stack.match(/:(\\d+):(\\d+)/); // fallback: first frame with line/col
            if (match) {
              const rawLine = Number(match[1]);
              // Our wrapped string puts user code starting at line USER_OFFSET + 1
              const userLine = Math.max(1, rawLine - USER_OFFSET);
              lineNum = userLine;
              const srcLines = (code ?? '').split('\\n');
              lineText = srcLines[userLine - 1]?.trim();
            }
          } catch {}
          if (!lineNum && err?.message) {
            const idMatch = String(err.message).match(/['"]?([A-Za-z_$][\\w$]*)['"]? is not defined/);
            if (idMatch) {
              const ident = idMatch[1];
              const srcLines = (code ?? '').split('\\n');
              const foundIdx = srcLines.findIndex((l) => l.includes(ident));
              if (foundIdx !== -1) {
                lineNum = foundIdx + 1;
                lineText = srcLines[foundIdx]?.trim();
              }
            }
          }
          const stack = err?.stack ? String(err.stack) : undefined;
          const firstStackFrame = stack?.split('\\n').find((l) => l.includes('<anonymous>'));
          send({
            type: 'error',
            message: err?.message ?? String(err),
            stack,
            frame: firstStackFrame,
            logs,
            line: lineNum,
            lineContent: lineText
          });
        }
      };

      self.onmessage = (e) => {
        run(e.data?.code ?? '');
      };
    `;

		const blob = new Blob([workerSource], { type: 'application/javascript' });
		const worker = new Worker(URL.createObjectURL(blob));

		const timer = setTimeout(() => {
			worker.terminate();
			resolve({ logs, error: 'Timed out' });
		}, timeoutMs);

		worker.onmessage = (event: MessageEvent) => {
			const { type } = event.data || {};
			if (type === 'log') {
				logs.push(event.data.text);
				return;
			}

			clearTimeout(timer);
			worker.terminate();

			if (type === 'error') {
				resolve({
					logs: event.data.logs ?? logs,
					error: event.data.message,
					errorLine: event.data.line,
					errorLineContent: event.data.lineContent,
					errorStack: event.data.stack,
					errorFrame: event.data.frame
				});
				return;
			}

			if (type === 'done') {
				const value = event.data.result;
				let rendered = '';
				try {
					rendered = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
				} catch {
					rendered = String(value);
				}
				resolve({ logs: event.data.logs ?? logs, result: rendered });
			}
		};

		worker.postMessage({ code });
	});
}

registerTool({
	name: CODE_INTERPRETER_JS_TOOL_NAME,
	label: 'Code Interpreter (JavaScript)',
	description: 'Run JavaScript in a sandboxed Worker and capture logs plus final value.',
	enableConfigKey: 'enableCodeInterpreterTool',
	defaultEnabled: false,
	definition: codeInterpreterToolDefinition,
	execute: async (argsJson: string) => {
		let code = argsJson;
		try {
			const parsedArgs = JSON.parse(argsJson);
			if (parsedArgs && typeof parsedArgs === 'object' && typeof parsedArgs.code === 'string') {
				code = parsedArgs.code;
			}
		} catch {
			// leave raw
		}
		const { result, logs, error, errorLine, errorLineContent, errorStack, errorFrame } =
			await runCodeInterpreter(code);
		let combined = '';
		if (logs?.length) combined += logs.join('\n');
		if (combined && (result !== undefined || error)) combined += '\n';
		if (error) {
			const lineLabel = errorLine !== undefined ? `line ${errorLine}` : null;
			const lineSnippet =
				errorLine !== undefined && errorLineContent ? `: ${errorLineContent.trim()}` : '';
			const lineInfo = lineLabel ? ` (${lineLabel}${lineSnippet})` : '';
			combined += `Error${lineInfo}: ${error}`;
			if (!lineLabel) {
				if (errorFrame) {
					combined += `\nFrame: ${errorFrame}`;
				} else if (errorStack) {
					combined += `\nStack: ${errorStack}`;
				}
			}
		} else if (result !== undefined) {
			combined += result;
		} else if (!combined) {
			combined = '(no output)';
		}
		return { content: combined, expression: code };
	}
});

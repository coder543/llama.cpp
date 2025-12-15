import type { ApiToolDefinition } from '$lib/types';

export interface ToolRegistration {
	name: string;
	label: string;
	description: string;
	enableConfigKey: string; // key in settings config
	defaultEnabled?: boolean;
	definition: ApiToolDefinition;
	execute: (argsJson: string) => Promise<{ content: string; expression?: string }>;
}

const tools: ToolRegistration[] = [];

export function registerTool(tool: ToolRegistration) {
	if (tools.find((t) => t.name === tool.name)) return;
	tools.push(tool);
}

export function getAllTools(): ToolRegistration[] {
	return tools;
}

export function getEnabledToolDefinitions(config: Record<string, any>): ApiToolDefinition[] {
	return tools.filter((t) => config[t.enableConfigKey]).map((t) => t.definition);
}

export function findToolByName(name: string): ToolRegistration | undefined {
	return tools.find((t) => t.name === name);
}

export function isToolEnabled(name: string, config: Record<string, any>): boolean {
	const tool = findToolByName(name);
	return !!(tool && config[tool.enableConfigKey]);
}

export function getToolSettingDefaults(): Record<string, boolean> {
	const defaults: Record<string, boolean> = {};
	for (const tool of tools) {
		defaults[tool.enableConfigKey] = tool.defaultEnabled ?? false;
	}
	return defaults;
}

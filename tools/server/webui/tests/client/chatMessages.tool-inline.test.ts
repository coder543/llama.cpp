import { describe, it, expect } from 'vitest';
import { render } from 'vitest-browser-svelte';
import { conversationsStore } from '$lib/stores/conversations.svelte';
import { settingsStore } from '$lib/stores/settings.svelte';
import { SETTING_CONFIG_DEFAULT } from '$lib/constants/settings-config';
import type { ChatRole, DatabaseMessage } from '$lib/types';
import TestMessagesWrapper from './components/TestMessagesWrapper.svelte';

// Utility to build a message quickly
const msg = (
	id: string,
	role: ChatRole,
	content: string,
	parent: string | null,
	extra: Partial<DatabaseMessage> = {}
): DatabaseMessage => ({
	id,
	convId: 'c1',
	type: 'text',
	role,
	content,
	thinking: '',
	toolCalls: '',
	parent: parent ?? '-1',
	children: [],
	timestamp: Date.now(),
	...extra
});

describe('ChatMessages inline tool rendering', () => {
	it('collapses reasoning+tool chain and shows arguments and result in one block', async () => {
		// Enable calculator tool (client-side tools)
		settingsStore.config = { ...SETTING_CONFIG_DEFAULT, enableCalculatorTool: true };

		// Conversation context
		conversationsStore.activeConversation = {
			id: 'c1',
			name: 'Test',
			currNode: null,
			lastModified: Date.now()
		};

		// Message chain: user -> assistant(thinking+toolcall) -> tool -> assistant(thinking) -> tool -> assistant(final)
		const user = msg('u1', 'user', 'Question', null);
		const a1 = msg('a1', 'assistant', '', user.id, {
			thinking: 'step1',
			toolCalls: JSON.stringify([
				{
					id: 'call-1',
					type: 'function',
					function: { name: 'calculator', arguments: JSON.stringify({ expression: '20.25/7.84' }) }
				}
			])
		});
		const t1 = msg(
			't1',
			'tool',
			JSON.stringify({ expression: '20.25/7.84', result: '2.5829', duration_ms: 1234 }),
			a1.id,
			{
				toolCallId: 'call-1'
			}
		);
		const a2 = msg('a2', 'assistant', '', t1.id, {
			thinking: 'step2',
			toolCalls: JSON.stringify([
				{
					id: 'call-2',
					type: 'function',
					function: {
						name: 'calculator',
						arguments: JSON.stringify({ expression: 'log2(2.5829)' })
					}
				}
			])
		});
		const t2 = msg(
			't2',
			'tool',
			JSON.stringify({ expression: 'log2(2.5829)', result: '1.3689', duration_ms: 50 }),
			a2.id,
			{
				toolCallId: 'call-2'
			}
		);
		const a3 = msg('a3', 'assistant', 'About 1.37 stops', t2.id, { thinking: 'final step' });

		const messages = [user, a1, t1, a2, t2, a3];
		conversationsStore.activeMessages = messages;

		const { container } = render(TestMessagesWrapper, {
			target: document.body,
			props: { messages }
		});

		// One assistant card after collapsing the chain
		const assistants = container.querySelectorAll('[aria-label="Assistant message with actions"]');
		expect(assistants.length).toBe(1);

		// Arguments and result should both be visible
		expect(container.textContent).toContain('Arguments');
		expect(container.textContent).toContain('20.25/7.84');
		expect(container.textContent).toContain('1.3689');
		expect(container.textContent).toContain('1.23s');
	});
});

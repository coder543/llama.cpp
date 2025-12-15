<script lang="ts">
	import { ChatMessage } from '$lib/components/app';
	import { chatStore } from '$lib/stores/chat.svelte';
	import { conversationsStore, activeConversation } from '$lib/stores/conversations.svelte';
	import { config } from '$lib/stores/settings.svelte';
	import { getMessageSiblings } from '$lib/utils';
	import { SvelteSet } from 'svelte/reactivity';
	import type {
		ApiChatCompletionToolCall,
		ChatMessageSiblingInfo,
		DatabaseMessage
	} from '$lib/types';

	interface Props {
		class?: string;
		messages?: DatabaseMessage[];
		onUserAction?: () => void;
	}

	let { class: className, messages = [], onUserAction }: Props = $props();

	// Prefer live store messages; fall back to the provided prop (e.g. initial render/tests).
	const sourceMessages = $derived(
		conversationsStore.activeMessages.length ? conversationsStore.activeMessages : messages
	);

	let allConversationMessages = $state<DatabaseMessage[]>([]);
	const currentConfig = config();

	function refreshAllMessages() {
		const conversation = activeConversation();

		if (conversation) {
			conversationsStore.getConversationMessages(conversation.id).then((messages) => {
				allConversationMessages = messages;
			});
		} else {
			allConversationMessages = [];
		}
	}

	// Single effect that tracks both conversation and message changes
	$effect(() => {
		const conversation = activeConversation();

		if (conversation) {
			refreshAllMessages();
		}
	});

	type ToolSegment =
		| { kind: 'thinking'; content: string }
		| { kind: 'tool'; toolCalls: ApiChatCompletionToolCall[]; parentId: string };
	type CollectedToolMessage = {
		toolCallId?: string | null;
		parsed: { expression?: string; result?: string; duration_ms?: number };
	};
	type AssistantDisplayMessage = DatabaseMessage & {
		_toolParentIds?: string[];
		_segments?: ToolSegment[];
		_toolMessagesCollected?: CollectedToolMessage[];
		_actionTargetId?: string;
	};
	type DisplayEntry = {
		message: DatabaseMessage | AssistantDisplayMessage;
		siblingInfo: ChatMessageSiblingInfo;
	};

	let displayMessages = $derived.by((): DisplayEntry[] => {
		// Force reactivity on message field changes (important for streaming updates)
		const signature = sourceMessages
			.map(
				(m) =>
					`${m.id}-${m.role}-${m.parent ?? ''}-${m.timestamp ?? ''}-${m.thinking ?? ''}-${
						m.toolCalls ?? ''
					}-${m.content ?? ''}`
			)
			.join('|');
		// signature is unused but ensures Svelte tracks the above fields
		void signature;

		if (!sourceMessages.length) return [];

		// Filter out system messages if showSystemMessage is false
		const filteredMessages = currentConfig.showSystemMessage
			? sourceMessages
			: sourceMessages.filter((msg) => msg.type !== 'system');

		const visited = new SvelteSet<string>();
		const result: DisplayEntry[] = [];

		const getChildren = (parentId: string, role?: string) =>
			filteredMessages.filter((m) => m.parent === parentId && (!role || m.role === role));

		const normalizeToolParsed = (
			value: unknown
		): { expression?: string; result?: string; duration_ms?: number } | null => {
			if (!value || typeof value !== 'object') return null;
			const obj = value as Record<string, unknown>;
			return {
				expression: typeof obj.expression === 'string' ? obj.expression : undefined,
				result: typeof obj.result === 'string' ? obj.result : undefined,
				duration_ms: typeof obj.duration_ms === 'number' ? obj.duration_ms : undefined
			};
		};

		for (const msg of filteredMessages) {
			if (visited.has(msg.id)) continue;
			// Don't render tools directly, but keep them for collection; skip marking visited here

			// Skip tool messages (rendered inline)
			if (msg.role === 'tool') continue;

			if (msg.role === 'assistant') {
				// Collapse consecutive assistant/tool chains into one display message
				const toolParentIds: string[] = [];
				const thinkingParts: string[] = [];
				const toolCallsCombined: ApiChatCompletionToolCall[] = [];
				const segments: ToolSegment[] = [];
				const toolMessagesCollected: CollectedToolMessage[] = [];
				const toolCallIds = new SvelteSet<string>();

				let currentAssistant: DatabaseMessage | undefined = msg;

				while (currentAssistant) {
					visited.add(currentAssistant.id);
					toolParentIds.push(currentAssistant.id);

					if (currentAssistant.thinking) {
						thinkingParts.push(currentAssistant.thinking);
						segments.push({ kind: 'thinking', content: currentAssistant.thinking });
					}
					let thisAssistantToolCalls: ApiChatCompletionToolCall[] = [];
					if (currentAssistant.toolCalls) {
						try {
							const parsed: unknown = JSON.parse(currentAssistant.toolCalls);
							if (Array.isArray(parsed)) {
								for (const tc of parsed as ApiChatCompletionToolCall[]) {
									if (tc?.id && toolCallIds.has(tc.id)) continue;
									if (tc?.id) toolCallIds.add(tc.id);
									toolCallsCombined.push(tc);
									thisAssistantToolCalls.push(tc);
								}
							}
						} catch {
							// ignore malformed
						}
					}
					if (thisAssistantToolCalls.length) {
						segments.push({
							kind: 'tool',
							toolCalls: thisAssistantToolCalls,
							parentId: currentAssistant.id
						});
					}

					const toolChildren = getChildren(currentAssistant.id, 'tool');
					for (const t of toolChildren) {
						visited.add(t.id);
						// capture parsed tool message for inline use
						try {
							const parsedUnknown: unknown = t.content ? JSON.parse(t.content) : null;
							const normalized = normalizeToolParsed(parsedUnknown);
							toolMessagesCollected.push({
								toolCallId: t.toolCallId,
								parsed: normalized ?? { result: t.content }
							});
						} catch {
							const p = { result: t.content };
							toolMessagesCollected.push({ toolCallId: t.toolCallId, parsed: p });
						}
					}

					// Assume at most one assistant child chained after tools
					let nextAssistant = toolChildren
						.map((t) => getChildren(t.id, 'assistant')[0])
						.find((a) => a !== undefined);

					// Also allow direct assistant->assistant continuation (no intervening tool)
					if (!nextAssistant) {
						nextAssistant = getChildren(currentAssistant.id, 'assistant')[0];
					}

					if (nextAssistant) {
						currentAssistant = nextAssistant;
						continue;
					}
					break;
				}

				const siblingInfo: ChatMessageSiblingInfo = getMessageSiblings(
					allConversationMessages,
					msg.id
				) || {
					message: msg,
					siblingIds: [msg.id],
					currentIndex: 0,
					totalSiblings: 1
				};

				const mergedAssistant: AssistantDisplayMessage = {
					...(currentAssistant ?? msg),
					content: currentAssistant?.content ?? '',
					thinking: thinkingParts.filter(Boolean).join('\n\n'),
					toolCalls: toolCallsCombined.length ? JSON.stringify(toolCallsCombined) : '',
					_toolParentIds: toolParentIds,
					_segments: segments,
					_actionTargetId: msg.id,
					_toolMessagesCollected: toolMessagesCollected
				};

				result.push({ message: mergedAssistant, siblingInfo });
				continue;
			}

			// user/system messages
			const siblingInfo: ChatMessageSiblingInfo = getMessageSiblings(
				allConversationMessages,
				msg.id
			) || {
				message: msg,
				siblingIds: [msg.id],
				currentIndex: 0,
				totalSiblings: 1
			};
			result.push({ message: msg, siblingInfo });
		}

		return result;
	});

	function getToolParentIdsForMessage(msg: DisplayEntry['message']): string[] | undefined {
		return (msg as AssistantDisplayMessage)._toolParentIds;
	}

	function getDisplayKeyForMessage(msg: DisplayEntry['message']): string {
		return (msg as AssistantDisplayMessage)._actionTargetId ?? msg.id;
	}

	async function handleNavigateToSibling(siblingId: string) {
		await conversationsStore.navigateToSibling(siblingId);
	}

	async function handleEditWithBranching(
		message: DatabaseMessage,
		newContent: string,
		newExtras?: DatabaseMessageExtra[]
	) {
		onUserAction?.();

		await chatStore.editMessageWithBranching(message.id, newContent, newExtras);

		refreshAllMessages();
	}

	async function handleEditWithReplacement(
		message: DatabaseMessage,
		newContent: string,
		shouldBranch: boolean
	) {
		onUserAction?.();

		await chatStore.editAssistantMessage(message.id, newContent, shouldBranch);

		refreshAllMessages();
	}

	async function handleRegenerateWithBranching(message: DatabaseMessage, modelOverride?: string) {
		onUserAction?.();

		await chatStore.regenerateMessageWithBranching(message.id, modelOverride);

		refreshAllMessages();
	}

	async function handleContinueAssistantMessage(message: DatabaseMessage) {
		onUserAction?.();

		await chatStore.continueAssistantMessage(message.id);

		refreshAllMessages();
	}

	async function handleEditUserMessagePreserveResponses(
		message: DatabaseMessage,
		newContent: string,
		newExtras?: DatabaseMessageExtra[]
	) {
		onUserAction?.();

		await chatStore.editUserMessagePreserveResponses(message.id, newContent, newExtras);

		refreshAllMessages();
	}

	async function handleDeleteMessage(message: DatabaseMessage) {
		await chatStore.deleteMessage(message.id);

		refreshAllMessages();
	}
</script>

<div class="flex h-full flex-col space-y-10 pt-16 md:pt-24 {className}" style="height: auto; ">
	{#each displayMessages as { message, siblingInfo } (getDisplayKeyForMessage(message))}
		<ChatMessage
			class="mx-auto w-full max-w-[48rem]"
			{message}
			{siblingInfo}
			toolParentIds={getToolParentIdsForMessage(message)}
			onDelete={handleDeleteMessage}
			onNavigateToSibling={handleNavigateToSibling}
			onEditWithBranching={handleEditWithBranching}
			onEditWithReplacement={handleEditWithReplacement}
			onEditUserMessagePreserveResponses={handleEditUserMessagePreserveResponses}
			onRegenerateWithBranching={handleRegenerateWithBranching}
			onContinueAssistantMessage={handleContinueAssistantMessage}
		/>
	{/each}
</div>

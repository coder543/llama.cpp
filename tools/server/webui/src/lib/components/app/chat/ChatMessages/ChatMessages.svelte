<script lang="ts">
	import { ChatMessage } from '$lib/components/app';
	import { chatStore } from '$lib/stores/chat.svelte';
	import { conversationsStore, activeConversation } from '$lib/stores/conversations.svelte';
	import { config } from '$lib/stores/settings.svelte';
	import { getMessageSiblings } from '$lib/utils';

	interface Props {
		class?: string;
		messages?: DatabaseMessage[];
		onUserAction?: () => void;
	}

	let { class: className, messages = [], onUserAction }: Props = $props();

	// Always react to live store messages; prop is used only for initial population.
	const sourceMessages = $derived(conversationsStore.activeMessages);

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

	let displayMessages = $derived.by(() => {
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
		signature;

		if (!sourceMessages.length) return [];

		// Filter out system messages if showSystemMessage is false
		const filteredMessages = currentConfig.showSystemMessage
			? sourceMessages
			: sourceMessages.filter((msg) => msg.type !== 'system');

		const idMap = new Map(filteredMessages.map((m) => [m.id, m]));
		const visited = new Set<string>();
		const result: {
			message: DatabaseMessage & { _toolParentIds?: string[]; _segments?: any[] };
			siblingInfo: any;
		}[] = [];

		const getChildren = (parentId: string, role?: string) =>
			filteredMessages.filter((m) => m.parent === parentId && (!role || m.role === role));

		for (const msg of filteredMessages) {
			if (visited.has(msg.id)) continue;
			// Don't render tools directly, but keep them for collection; skip marking visited here

			// Skip tool messages (rendered inline)
			if (msg.role === 'tool') continue;

			if (msg.role === 'assistant') {
				// Collapse consecutive assistant/tool chains into one display message
				const toolParentIds: string[] = [];
				const thinkingParts: string[] = [];
				const toolCallsCombined: any[] = [];
				const segments: any[] = [];
				const toolMessagesCollected: { toolCallId?: string | null; parsed: any }[] = [];
				const toolCallIds = new Set<string>();

				let currentAssistant: DatabaseMessage | undefined = msg;

				while (currentAssistant) {
					visited.add(currentAssistant.id);
					toolParentIds.push(currentAssistant.id);

					if (currentAssistant.thinking) {
						thinkingParts.push(currentAssistant.thinking);
						segments.push({ kind: 'thinking', content: currentAssistant.thinking });
					}
					let thisAssistantToolCalls: any[] = [];
					if (currentAssistant.toolCalls) {
						try {
							const parsed = JSON.parse(currentAssistant.toolCalls);
							if (Array.isArray(parsed)) {
								parsed.forEach((tc) => {
									if (tc?.id && toolCallIds.has(tc.id)) {
										return;
									}
									if (tc?.id) toolCallIds.add(tc.id);
									toolCallsCombined.push(tc);
									thisAssistantToolCalls.push(tc);
								});
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
							const parsed = t.content ? JSON.parse(t.content) : null;
							if (parsed && typeof parsed === 'object') {
								toolMessagesCollected.push({ toolCallId: t.toolCallId, parsed });
							} else {
								const p = { result: t.content };
								toolMessagesCollected.push({ toolCallId: t.toolCallId, parsed: p });
							}
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

				const siblingInfo = getMessageSiblings(allConversationMessages, msg.id) || {
					message: msg,
					siblingIds: [msg.id],
					currentIndex: 0,
					totalSiblings: 1
				};

				const mergedAssistant: DatabaseMessage & {
					_toolParentIds?: string[];
					_segments?: any[];
					_actionTargetId?: string;
				} = {
					...currentAssistant!,
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
			const siblingInfo = getMessageSiblings(allConversationMessages, msg.id) || {
				message: msg,
				siblingIds: [msg.id],
				currentIndex: 0,
				totalSiblings: 1
			};
			result.push({ message: msg as any, siblingInfo });
		}

		return result;
	});

	$effect(() => {
		if (typeof window !== 'undefined') {
			(window as any).__debugActiveMessages = sourceMessages;
			(window as any).__debugDisplayMessages = displayMessages;
		}
	});

	async function handleNavigateToSibling(siblingId: string) {
		await conversationsStore.navigateToSibling(siblingId);
	}

	async function handleEditWithBranching(message: DatabaseMessage, newContent: string) {
		onUserAction?.();

		await chatStore.editMessageWithBranching(message.id, newContent);

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
		newContent: string
	) {
		onUserAction?.();

		await chatStore.editUserMessagePreserveResponses(message.id, newContent);

		refreshAllMessages();
	}

	async function handleDeleteMessage(message: DatabaseMessage) {
		await chatStore.deleteMessage(message.id);

		refreshAllMessages();
	}
</script>

<div class="flex h-full flex-col space-y-10 pt-16 md:pt-24 {className}" style="height: auto; ">
	{#each displayMessages as { message, siblingInfo } (message._actionTargetId ?? message.id)}
		<ChatMessage
			class="mx-auto w-full max-w-[48rem]"
			{message}
			{siblingInfo}
			toolParentIds={(message as any)._toolParentIds}
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

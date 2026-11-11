import type { ProviderAdapter } from '../../types/provider-adapter';

/**
 * reasoning_content adapter
 *
 * Handles providers (e.g. Fireworks) that use `reasoning_content` instead of
 * the `reasoning` field name used by mainstream OpenAI-compatible clients.
 *
 * Outbound (preDispatch):
 *   - On assistant messages: `reasoning` → `reasoning_content`
 *   - On assistant messages: `thinking.content` → `reasoning_content` (if no `reasoning` present)
 *   - Option `fillMissing: true` additionally guarantees `reasoning_content` on
 *     assistant messages that carry no reasoning at all, sending `''`. Providers
 *     such as DeepSeek in thinking mode reject the whole request when any
 *     assistant turn lacks the field (400 "The `reasoning_content` in the
 *     thinking mode must be passed back to the API"), which happens whenever
 *     earlier turns in the history were produced by a different provider.
 *
 * Inbound (postDispatch):
 *   - On response choices: `reasoning_content` → `reasoning`
 *
 * Stream (preDispatchStreamChunk):
 *   - Rewrites `"reasoning_content":` → `"reasoning":` in raw SSE lines
 *   - This covers both delta fields in streaming chunks
 *
 * Stream (postDispatchStreamChunk):
 *   - Not needed: by the time postDispatchStreamChunk runs the provider has
 *     already emitted `reasoning_content`; preDispatchStreamChunk handles it.
 */
export const reasoningContentAdapter: ProviderAdapter = {
  name: 'reasoning_content',

  preDispatch(payload: Record<string, any>, options?: Record<string, any>): Record<string, any> {
    if (!Array.isArray(payload.messages)) return payload;

    const fillMissing = options?.fillMissing === true;

    const messages = payload.messages.map((msg: any) => {
      if (msg.role !== 'assistant') return msg;

      // Priority: explicit `reasoning` field first, then `thinking.content`.
      // `null` counts as absent: a provider that demands the passback rejects a null
      // reasoner just as hard as a missing one.
      const reasoningValue =
        msg.reasoning != null
          ? msg.reasoning
          : msg.thinking?.content != null
            ? msg.thinking.content
            : undefined;

      if (reasoningValue === undefined) {
        // Nothing to rename. Leave the message untouched unless `fillMissing` is
        // enabled, so providers that tolerate an absent reasoning field receive
        // exactly what the client sent.
        if (!fillMissing) return msg;

        // Preserve a value the client already supplied; only substitute when the
        // field is absent (or null). Legacy keys are stripped exactly as in the
        // rename branch below -- providers that reject `messages[N].reasoning`
        // would otherwise reject the payload the adapter exists to repair.
        const { reasoning: _r, thinking: _t, ...rest } = msg;
        return { ...rest, reasoning_content: msg.reasoning_content ?? '' };
      }

      const { reasoning: _r, thinking: _t, ...rest } = msg;
      return { ...rest, reasoning_content: reasoningValue };
    });

    return { ...payload, messages };
  },

  postDispatch(response: Record<string, any>, _options?: Record<string, any>): Record<string, any> {
    // The OpenAI transformer's transformResponse() already reads
    // `message.reasoning_content` natively, so no field rename is needed
    // on the inbound path. This is intentionally a no-op.
    return response;
  },

  preDispatchStreamChunk(line: string, _options?: Record<string, any>): string {
    // Replace `"reasoning":` with `"reasoning_content":` in the JSON payload
    // of SSE data lines so the provider receives the correct field name.
    if (!line.startsWith('data:')) return line;
    return line.replace(/"reasoning":/g, '"reasoning_content":');
  },
};

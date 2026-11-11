import { describe, expect, it } from 'bun:test';

import { reasoningContentAdapter } from '../../transformers/adapters/reasoning-content.adapter';

describe('reasoningContentAdapter', () => {
  it('has the correct name', () => {
    expect(reasoningContentAdapter.name).toBe('reasoning_content');
  });

  // ── preDispatch ──────────────────────────────────────────────────────────

  describe('preDispatch', () => {
    it('renames reasoning → reasoning_content on assistant messages', () => {
      const payload = {
        messages: [
          { role: 'user', content: 'hello' },
          { role: 'assistant', content: 'hi', reasoning: 'I thought about it' },
        ],
      };
      const result = reasoningContentAdapter.preDispatch(payload);
      expect(result.messages[1]).toEqual({
        role: 'assistant',
        content: 'hi',
        reasoning_content: 'I thought about it',
      });
      expect(result.messages[1].reasoning).toBeUndefined();
    });

    it('uses thinking.content when reasoning is absent', () => {
      const payload = {
        messages: [
          {
            role: 'assistant',
            content: 'hi',
            thinking: { content: 'deep thought' },
          },
        ],
      };
      const result = reasoningContentAdapter.preDispatch(payload);
      expect(result.messages[0]).toEqual({
        role: 'assistant',
        content: 'hi',
        reasoning_content: 'deep thought',
      });
      expect(result.messages[0].thinking).toBeUndefined();
    });

    it('prefers reasoning over thinking.content when both present', () => {
      const payload = {
        messages: [
          {
            role: 'assistant',
            content: 'hi',
            reasoning: 'explicit reasoning',
            thinking: { content: 'thinking block' },
          },
        ],
      };
      const result = reasoningContentAdapter.preDispatch(payload);
      expect(result.messages[0].reasoning_content).toBe('explicit reasoning');
    });

    it('does not modify non-assistant messages', () => {
      const payload = {
        messages: [{ role: 'user', content: 'hello', reasoning: 'should stay' }],
      };
      const result = reasoningContentAdapter.preDispatch(payload);
      expect(result.messages[0].reasoning).toBe('should stay');
      expect(result.messages[0].reasoning_content).toBeUndefined();
    });

    it('is a no-op when messages have no reasoning fields', () => {
      const payload = {
        messages: [{ role: 'assistant', content: 'hi' }],
      };
      const result = reasoningContentAdapter.preDispatch(payload);
      expect(result.messages[0]).toEqual({ role: 'assistant', content: 'hi' });
    });

    it('returns payload unchanged when messages is missing', () => {
      const payload = { model: 'foo' };
      expect(reasoningContentAdapter.preDispatch(payload)).toEqual({ model: 'foo' });
    });

    describe('fillMissing option', () => {
      it('fills absent reasoning_content with an empty string when enabled', () => {
        // Mirrors the DeepSeek thinking-mode rejection: a history where some
        // assistant turns carry reasoning and others carry none at all.
        const payload = {
          tools: [{ type: 'function', function: { name: 'noop' } }],
          messages: [
            { role: 'system', content: 'sys' },
            { role: 'user', content: 'go' },
            { role: 'assistant', content: '', tool_calls: [{ id: 'c1' }] },
            { role: 'tool', tool_call_id: 'c1', content: 'ok' },
            { role: 'assistant', content: '', reasoning: 'real thought' },
          ],
        };

        const result = reasoningContentAdapter.preDispatch(payload, { fillMissing: true });

        const assistants = result.messages.filter((m: any) => m.role === 'assistant');
        expect(assistants).toHaveLength(2);
        for (const message of assistants) {
          expect(typeof message.reasoning_content).toBe('string');
        }
        // missing reasoning becomes an empty string
        expect(assistants[0].reasoning_content).toBe('');
        // real reasoning is preserved, not overwritten, and the legacy field is removed
        expect(assistants[1].reasoning_content).toBe('real thought');
        expect(assistants[1].reasoning).toBeUndefined();
        // non-assistant messages are untouched
        expect(result.messages[0]).toEqual({ role: 'system', content: 'sys' });
        expect(result.messages[3]).toEqual({ role: 'tool', tool_call_id: 'c1', content: 'ok' });
      });

      it('is off by default and leaves absent reasoning_content absent', () => {
        const payload = { messages: [{ role: 'assistant', content: 'hi' }] };
        const result = reasoningContentAdapter.preDispatch(payload);
        expect('reasoning_content' in result.messages[0]).toBe(false);
      });

      it('treats a null reasoner as missing and fills it', () => {
        const payload = {
          messages: [
            { role: 'assistant', content: '', reasoning: null },
            { role: 'assistant', content: '', reasoning_content: null },
          ],
        };

        const result = reasoningContentAdapter.preDispatch(payload, { fillMissing: true });

        expect(result.messages[0].reasoning_content).toBe('');
        expect(result.messages[1].reasoning_content).toBe('');
        // the legacy fields must not survive: providers that reject
        // `messages[N].reasoning` would fail on them just as before
        expect(result.messages[0].reasoning).toBeUndefined();
        expect(result.messages[1].reasoning).toBeUndefined();
      });
    });
  });

  // ── postDispatch ─────────────────────────────────────────────────────────

  describe('postDispatch', () => {
    it('is a no-op — the OpenAI transformer already reads reasoning_content natively', () => {
      const response = {
        choices: [
          {
            message: { role: 'assistant', content: 'hi', reasoning_content: 'thought' },
            finish_reason: 'stop',
          },
        ],
      };
      const result = reasoningContentAdapter.postDispatch(response);
      // reasoning_content must be preserved as-is for transformResponse() to pick up
      expect(result.choices[0].message.reasoning_content).toBe('thought');
      expect(result.choices[0].message.reasoning).toBeUndefined();
    });

    it('returns response unchanged when choices is missing', () => {
      const response = { id: 'foo' };
      expect(reasoningContentAdapter.postDispatch(response)).toEqual({ id: 'foo' });
    });
  });

  // ── preDispatchStreamChunk ───────────────────────────────────────────────

  describe('preDispatchStreamChunk', () => {
    it('rewrites "reasoning": to "reasoning_content": in data lines', () => {
      const line = 'data: {"delta":{"reasoning":"thought"}}';
      const result = reasoningContentAdapter.preDispatchStreamChunk!(line);
      expect(result).toBe('data: {"delta":{"reasoning_content":"thought"}}');
    });

    it('does not modify non-data lines', () => {
      const line = 'event: message_start';
      expect(reasoningContentAdapter.preDispatchStreamChunk!(line)).toBe(line);
    });

    it('handles multiple occurrences in one line', () => {
      const line = 'data: {"a":{"reasoning":"x"},"b":{"reasoning":"y"}}';
      const result = reasoningContentAdapter.preDispatchStreamChunk!(line);
      expect(result).toBe('data: {"a":{"reasoning_content":"x"},"b":{"reasoning_content":"y"}}');
    });
  });
});

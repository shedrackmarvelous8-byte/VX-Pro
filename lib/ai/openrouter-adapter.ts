import type { StandardChatMessage } from './context'

export interface OpenRouterGenerationResult {
  text: string
  usage?: {
    promptTokens: number
    completionTokens: number
    totalTokens: number
  }
}

function getOpenRouterApiKey(): string {
  const key = process.env.OPENROUTER_API_KEY
  if (!key) {
    throw new Error('OPENROUTER_API_KEY is not configured on the server')
  }
  return key
}

function formatMessagesForOpenRouter(systemPrompt: string, messages: StandardChatMessage[]) {
  const formatted: any[] = [{ role: 'system', content: systemPrompt }]
  for (const m of messages) {
    if (m.images && m.images.length > 0) {
      formatted.push({
        role: m.role,
        content: [
          { type: 'text', text: m.content || ' ' },
          ...m.images.map((img) => ({
            type: 'image_url',
            image_url: { url: `data:${img.mimeType};base64,${img.data}` },
          })),
        ],
      })
    } else {
      formatted.push({ role: m.role, content: m.content })
    }
  }
  return formatted
}

/**
 * Calculates a safe, reasonable max completion token limit based on prompt length and task type.
 * Avoids requesting universal 65,536 tokens which triggers OpenRouter credit affordability errors.
 */
export function calculateSafeMaxTokens(params?: {
  prompt?: string
  isCodingTask?: boolean
  customMaxTokens?: number
  modelContextWindow?: number
}): number {
  if (params?.customMaxTokens && params.customMaxTokens > 0 && params.customMaxTokens <= 16384) {
    return params.customMaxTokens
  }

  const clean = (params?.prompt || '').trim().toLowerCase()
  const isShortGreeting =
    clean.length < 30 &&
    /^(hi|hello|hey|thanks|thank you|ok|okay|cool|good|who are you|what is vx)\b/i.test(clean)

  if (isShortGreeting) {
    return 1024
  }

  if (params?.isCodingTask) {
    return 6144
  }

  return 2048
}

/**
 * Translates raw OpenRouter provider HTTP/JSON error responses into human-friendly VX error messages.
 */
function parseOpenRouterError(status: number, errorBody: any, modelName: string): Error {
  const rawMsg = errorBody?.error?.message || errorBody?.message || ''
  const lowerMsg = rawMsg.toLowerCase()

  // 1. Credit & token affordability limit (402 or balance error)
  if (
    status === 402 ||
    lowerMsg.includes('afford') ||
    lowerMsg.includes('credit') ||
    lowerMsg.includes('balance') ||
    lowerMsg.includes('insufficient')
  ) {
    return new Error(
      'Your selected model needs more available credits for this request. Try a shorter request, choose another available model, or use Auto.'
    )
  }

  // 2. High demand / 503 Service Unavailable / Overloaded
  if (
    status === 503 ||
    status === 429 ||
    lowerMsg.includes('high demand') ||
    lowerMsg.includes('overloaded') ||
    lowerMsg.includes('service unavailable') ||
    lowerMsg.includes('rate limit')
  ) {
    return new Error(
      `${modelName} is currently experiencing high demand (503). You can try again or switch to another available model or Auto.`
    )
  }

  // 3. Model not found or unavailable
  if (status === 404 || lowerMsg.includes('not found') || lowerMsg.includes('disabled')) {
    return new Error(
      `The model ${modelName} is currently unavailable from OpenRouter. Please choose another model or use Auto.`
    )
  }

  // 4. Authentication error
  if (status === 401 || lowerMsg.includes('api key') || lowerMsg.includes('unauthorized')) {
    return new Error('OpenRouter service authentication failed. Please verify server API key configuration.')
  }

  return new Error(rawMsg || `OpenRouter returned status code ${status}`)
}

export async function generateOpenRouterResponse(params: {
  targetModel: string
  systemPrompt: string
  messages: StandardChatMessage[]
  prompt?: string
  isCodingTask?: boolean
  maxTokens?: number
}): Promise<OpenRouterGenerationResult> {
  const apiKey = getOpenRouterApiKey()
  const formattedMessages = formatMessagesForOpenRouter(params.systemPrompt, params.messages)
  const lastUserMessage = params.prompt || params.messages.filter((m) => m.role === 'user').pop()?.content || ''
  const maxTokens = calculateSafeMaxTokens({
    prompt: lastUserMessage,
    isCodingTask: params.isCodingTask,
    customMaxTokens: params.maxTokens,
  })

  const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      'HTTP-Referer': 'https://vx.dev',
      'X-Title': 'VX Development Workspace',
    },
    body: JSON.stringify({
      model: params.targetModel,
      messages: formattedMessages,
      max_tokens: maxTokens,
      stream: false,
    }),
  })

  if (!response.ok) {
    const errorBody = await response.json().catch(() => null)
    throw parseOpenRouterError(response.status, errorBody, params.targetModel)
  }

  const data = await response.json()
  const text = data?.choices?.[0]?.message?.content || ''
  const usage = data?.usage
    ? {
        promptTokens: data.usage.prompt_tokens || 0,
        completionTokens: data.usage.completion_tokens || 0,
        totalTokens: data.usage.total_tokens || 0,
      }
    : undefined

  return { text, usage }
}

export async function streamOpenRouterResponse(params: {
  targetModel: string
  systemPrompt: string
  messages: StandardChatMessage[]
  prompt?: string
  isCodingTask?: boolean
  maxTokens?: number
  onChunk: (chunk: string) => Promise<void> | void
}): Promise<OpenRouterGenerationResult> {
  const apiKey = getOpenRouterApiKey()
  const formattedMessages = formatMessagesForOpenRouter(params.systemPrompt, params.messages)
  const lastUserMessage = params.prompt || params.messages.filter((m) => m.role === 'user').pop()?.content || ''
  const maxTokens = calculateSafeMaxTokens({
    prompt: lastUserMessage,
    isCodingTask: params.isCodingTask,
    customMaxTokens: params.maxTokens,
  })

  const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      'HTTP-Referer': 'https://vx.dev',
      'X-Title': 'VX Development Workspace',
    },
    body: JSON.stringify({
      model: params.targetModel,
      messages: formattedMessages,
      max_tokens: maxTokens,
      stream: true,
    }),
  })

  if (!response.ok) {
    const errorBody = await response.json().catch(() => null)
    throw parseOpenRouterError(response.status, errorBody, params.targetModel)
  }

  if (!response.body) {
    throw new Error('No response body received from OpenRouter')
  }

  let fullText = ''
  let finalUsage: OpenRouterGenerationResult['usage'] = undefined

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() || ''

    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed || !trimmed.startsWith('data:')) continue

      const dataStr = trimmed.replace(/^data:\s*/, '')
      if (dataStr === '[DONE]') continue

      try {
        const parsed = JSON.parse(dataStr)
        const delta = parsed.choices?.[0]?.delta?.content
        if (delta) {
          fullText += delta
          await params.onChunk(delta)
        }

        if (parsed.usage) {
          finalUsage = {
            promptTokens: parsed.usage.prompt_tokens || 0,
            completionTokens: parsed.usage.completion_tokens || 0,
            totalTokens: parsed.usage.total_tokens || 0,
          }
        }
      } catch {
        // Skip unparseable lines
      }
    }
  }

  return { text: fullText, usage: finalUsage }
}

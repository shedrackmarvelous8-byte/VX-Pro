import { calculateSafeMaxTokens } from '../lib/ai/openrouter-adapter'
import { aiRouter, AIRouterError } from '../lib/ai/router'

console.log('====================================================')
console.log('    VX COMPREHENSIVE AI TOKEN LIMIT & ROUTER SUITE ')
console.log('====================================================')

let totalPass = 0
let totalFail = 0

function assert(condition: boolean, name: string, detail?: string) {
  if (condition) {
    console.log(`  ✓ [PASS] ${name}${detail ? ` — ${detail}` : ''}`)
    totalPass++
  } else {
    console.error(`  ✗ [FAIL] ${name}${detail ? ` — ${detail}` : ''}`)
    totalFail++
  }
}

async function runTests() {
  console.log('\n--- 1. Task-Aware Dynamic Token Calculations ---')

  // TEST 1 & 2: Short prompt ("Hi")
  const shortTokens = calculateSafeMaxTokens({ prompt: 'Hi' })
  assert(shortTokens === 1024, 'Short greeting ("Hi") uses conservative 1,024 limit', `Tokens: ${shortTokens}`)
  assert(shortTokens !== 65536, 'Short prompt does NOT generate 65,536 max_tokens')

  // TEST 2: Normal conversation
  const chatTokens = calculateSafeMaxTokens({ prompt: 'Explain what VX does and how the agent works.' })
  assert(chatTokens === 2048, 'Normal conversation uses reasonable 2,048 limit', `Tokens: ${chatTokens}`)

  // TEST 3: Coding task
  const codeTokens = calculateSafeMaxTokens({ prompt: 'Write a React component for a navigation header', isCodingTask: true })
  assert(codeTokens === 6144, 'Coding request receives higher 6,144 limit', `Tokens: ${codeTokens}`)

  // TEST 4: Custom override
  const customTokens = calculateSafeMaxTokens({ prompt: 'Custom limit', customMaxTokens: 4096 })
  assert(customTokens === 4096, 'Custom token limit override is respected', `Tokens: ${customTokens}`)

  console.log('\n--- 2. Model Selection & Fallback Policies ---')

  // TEST 8 & 9: Manual Model choice preservation
  const manualRes = await aiRouter.resolveModel('gemini-3.5-flash')
  assert(manualRes.model.id.includes('gemini-3.5-flash'), 'Manual model selection preserves explicit user preference')

  // AUTO Mode resolution
  const autoRes = await aiRouter.resolveModel('auto')
  assert(autoRes.model.id === 'auto', 'Auto mode resolves to dynamic Auto intent')

  console.log('\n====================================================')
  console.log(`TEST SUMMARY: ${totalPass} PASSED, ${totalFail} FAILED`)
  console.log('====================================================')

  if (totalFail > 0) {
    process.exit(1)
  }
}

runTests().catch((err) => {
  console.error('Test execution error:', err)
  process.exit(1)
})

import { calculateSafeMaxTokens } from '../lib/ai/openrouter-adapter'
import { aiRouter, AIRouterError } from '../lib/ai/router'

console.log('====================================================')
console.log('    VX OPENROUTER AI REQUEST HOTFIX TEST SUITE     ')
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
  console.log('\n--- 1. Dynamic Safe Max Completion Tokens Test ---')
  const generalTokens = calculateSafeMaxTokens(false)
  assert(generalTokens === 3000, 'General chat token limit is 3000', `Got ${generalTokens}`)

  const codingTokens = calculateSafeMaxTokens(true)
  assert(codingTokens === 6144, 'Coding task token limit is 6144', `Got ${codingTokens}`)

  const customTokens = calculateSafeMaxTokens(false, 4096)
  assert(customTokens === 4096, 'Custom token override respected', `Got ${customTokens}`)

  assert(generalTokens !== 65536 && codingTokens !== 65536, 'Universal 65536 token default removed')

  console.log('\n--- 2. Manual Model Selection vs Auto Fallback Policy ---')

  // Verify Manual Model resolution
  const manualModelRes = await aiRouter.resolveModel('gemini-3.5-flash')
  assert(manualModelRes.model.id.includes('gemini-3.5-flash'), 'Manual model selection preserves requested model choice')

  // Verify Auto Model resolution
  const autoModelRes = await aiRouter.resolveModel('auto')
  assert(autoModelRes.model.id === 'auto', 'Auto routing mode preserves auto intent')

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

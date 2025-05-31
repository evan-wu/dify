import type { CommonNodeType, ValueSelector } from '@/app/components/workflow/types'

export enum RaceStrategy {
  FIRST_COMPLETE = 'first_complete',
  FASTEST_VALID = 'fastest_valid',
  TIMEOUT_BEST = 'timeout_best',
  QUALITY_RACE = 'quality_race',
}

export enum WinCondition {
  ANY_RESULT = 'any_result',
  NO_ERROR = 'no_error',
  CUSTOM_VALIDATION = 'custom_validation',
}

export type RaceNodeType = CommonNodeType & {
  race_strategy: RaceStrategy
  win_condition: WinCondition
  timeout_seconds: number
  max_winners: number
  variables: ValueSelector[]
  validation_expression?: string
  scoring_expression?: string
  fail_on_timeout: boolean
  fail_on_all_errors: boolean
} 
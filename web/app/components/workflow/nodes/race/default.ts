import { type NodeDefault } from '../../types'
import { BlockEnum } from '../../types'
import type { RaceNodeType } from './types'
import { RaceStrategy, WinCondition } from './types'
import { ALL_CHAT_AVAILABLE_BLOCKS, ALL_COMPLETION_AVAILABLE_BLOCKS } from '@/app/components/workflow/blocks'

const i18nPrefix = 'workflow'

const nodeDefault: NodeDefault<RaceNodeType> = {
  defaultValue: {
    race_strategy: RaceStrategy.FIRST_COMPLETE,
    win_condition: WinCondition.ANY_RESULT,
    timeout_seconds: 30,
    max_winners: 1,
    variables: [],
    fail_on_timeout: false,
    fail_on_all_errors: true,
  },
  getAvailablePrevNodes(isChatMode: boolean) {
    const nodes = isChatMode
      ? ALL_CHAT_AVAILABLE_BLOCKS
      : ALL_COMPLETION_AVAILABLE_BLOCKS.filter(type => type !== BlockEnum.End)
    return nodes
  },
  getAvailableNextNodes(isChatMode: boolean) {
    const nodes = isChatMode ? ALL_CHAT_AVAILABLE_BLOCKS : ALL_COMPLETION_AVAILABLE_BLOCKS
    return nodes
  },
  checkValid(payload: RaceNodeType, t: any) {
    let errorMessages = ''
    const { variables, timeout_seconds, max_winners } = payload

    // Check if variables are configured
    if (!variables || variables.length === 0) {
      errorMessages = t(`${i18nPrefix}.errorMsg.fieldRequired`, { 
        field: t(`${i18nPrefix}.nodes.race.variables`) 
      })
    }

    // Check if variables are valid (not empty)
    if (!errorMessages && variables) {
      variables.forEach((variable, index) => {
        if (!variable || variable.length === 0) {
          errorMessages = t(`${i18nPrefix}.errorMsg.fieldRequired`, { 
            field: t(`${i18nPrefix}.nodes.race.variable`) + ` ${index + 1}` 
          })
        }
      })
    }

    // Check timeout is positive
    if (!errorMessages && (!timeout_seconds || timeout_seconds <= 0)) {
      errorMessages = t(`${i18nPrefix}.errorMsg.fieldRequired`, { 
        field: t(`${i18nPrefix}.nodes.race.timeout`) 
      })
    }

    // Check max_winners is positive
    if (!errorMessages && (!max_winners || max_winners <= 0)) {
      errorMessages = t(`${i18nPrefix}.errorMsg.fieldRequired`, { 
        field: t(`${i18nPrefix}.nodes.race.maxWinners`) 
      })
    }

    // Check if we have at least 2 variables for racing to make sense
    if (!errorMessages && variables && variables.length < 2) {
      errorMessages = t(`${i18nPrefix}.errorMsg.atLeastTwo`, { 
        field: t(`${i18nPrefix}.nodes.race.variables`) 
      })
    }

    return {
      isValid: !errorMessages,
      errorMessage: errorMessages,
    }
  },
}

export default nodeDefault 
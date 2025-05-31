import { useCallback } from 'react'
import produce from 'immer'
import type { RaceNodeType } from './types'
import { RaceStrategy, WinCondition } from './types'
import type { ValueSelector, Var } from '@/app/components/workflow/types'
import { BlockEnum, VarType } from '@/app/components/workflow/types'
import { useNodesReadOnly } from '@/app/components/workflow/hooks'
import useNodeCrud from '@/app/components/workflow/nodes/_base/hooks/use-node-crud'
import useAvailableVarList from '@/app/components/workflow/nodes/_base/hooks/use-available-var-list'

const useConfig = (id: string, payload: RaceNodeType) => {
  const { nodesReadOnly: readOnly } = useNodesReadOnly()
  const { inputs, setInputs } = useNodeCrud<RaceNodeType>(id, payload)
  const { availableVars, availableNodesWithParent } = useAvailableVarList(id, {
    onlyLeafNodeVar: false,
    filterVar: (variable: Var) => {
      return true // Accept all variable types for racing
    },
  })

  const filterVar = useCallback((variable: Var) => {
    return true // Accept all variable types for racing
  }, [])

  const handleRaceStrategyChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    const newStrategy = e.target.value as RaceStrategy
    const newInputs = produce(inputs, (draft) => {
      draft.race_strategy = newStrategy
    })
    setInputs(newInputs)
  }, [inputs, setInputs])

  const handleWinConditionChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    const newCondition = e.target.value as WinCondition
    const newInputs = produce(inputs, (draft) => {
      draft.win_condition = newCondition
    })
    setInputs(newInputs)
  }, [inputs, setInputs])

  const handleTimeoutChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const timeout = parseInt(e.target.value, 10)
    const newInputs = produce(inputs, (draft) => {
      draft.timeout_seconds = timeout || 30
    })
    setInputs(newInputs)
  }, [inputs, setInputs])

  const handleMaxWinnersChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const maxWinners = parseInt(e.target.value, 10)
    const newInputs = produce(inputs, (draft) => {
      draft.max_winners = maxWinners || 1
    })
    setInputs(newInputs)
  }, [inputs, setInputs])

  const handleFailOnTimeoutChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const newInputs = produce(inputs, (draft) => {
      draft.fail_on_timeout = e.target.checked
    })
    setInputs(newInputs)
  }, [inputs, setInputs])

  const handleFailOnAllErrorsChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const newInputs = produce(inputs, (draft) => {
      draft.fail_on_all_errors = e.target.checked
    })
    setInputs(newInputs)
  }, [inputs, setInputs])

  const handleValidationExpressionChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newInputs = produce(inputs, (draft) => {
      draft.validation_expression = e.target.value
    })
    setInputs(newInputs)
  }, [inputs, setInputs])

  const handleScoringExpressionChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newInputs = produce(inputs, (draft) => {
      draft.scoring_expression = e.target.value
    })
    setInputs(newInputs)
  }, [inputs, setInputs])

  const handleVariableChange = useCallback((index: number, variable: ValueSelector) => {
    const newInputs = produce(inputs, (draft) => {
      draft.variables[index] = variable
    })
    setInputs(newInputs)
  }, [inputs, setInputs])

  const handleVariableAdd = useCallback(() => {
    const newInputs = produce(inputs, (draft) => {
      if (!draft.variables) {
        draft.variables = []
      }
      draft.variables.push([])
    })
    setInputs(newInputs)
  }, [inputs, setInputs])

  const handleVariableRemove = useCallback((index: number) => {
    const newInputs = produce(inputs, (draft) => {
      draft.variables.splice(index, 1)
    })
    setInputs(newInputs)
  }, [inputs, setInputs])

  return {
    readOnly,
    inputs,
    handleRaceStrategyChange,
    handleWinConditionChange,
    handleTimeoutChange,
    handleMaxWinnersChange,
    handleFailOnTimeoutChange,
    handleFailOnAllErrorsChange,
    handleValidationExpressionChange,
    handleScoringExpressionChange,
    handleVariableChange,
    handleVariableAdd,
    handleVariableRemove,
    filterVar,
    availableVars,
    availableNodesWithParent,
  }
}

export default useConfig 
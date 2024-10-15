import { useCallback } from 'react'
import produce from 'immer'
import { useBoolean } from 'ahooks'
import {
  useIsChatMode,
  useIsNodeInIteration,
  useNodesReadOnly,
  useWorkflow,
} from '../../hooks'
import { VarType } from '../../types'
import type { ValueSelector, Var } from '../../types'
import useNodeCrud from '../_base/hooks/use-node-crud'
import { getNodeInfoById, getNodeUsedVarPassToServerKey, getNodeUsedVars, isSystemVar, toNodeOutputVars } from '../_base/components/variable/utils'
import useOneStepRun from '../_base/hooks/use-one-step-run'
import type { CollectNodeType } from './types'
import type { VarType as VarKindType } from '@/app/components/workflow/nodes/tool/types'
import useAvailableVarList from "@/app/components/workflow/nodes/_base/hooks/use-available-var-list";
import {
  HandleAddCondition,
  HandleRemoveCondition,
  HandleUpdateCondition, HandleUpdateConditionLogicalOperator
} from "@/app/components/workflow/nodes/if-else/types";
import { v4 as uuid4 } from "uuid";
import { getOperators } from "@/app/components/workflow/nodes/if-else/utils";

const DELIMITER = '@@@@@'
const useConfig = (id: string, payload: CollectNodeType) => {
  const { nodesReadOnly: readOnly } = useNodesReadOnly()
  const { isNodeInIteration } = useIsNodeInIteration(id)
  const isChatMode = useIsChatMode()

  const { inputs, setInputs } = useNodeCrud<CollectNodeType>(id, payload)

  const filterVar = useCallback((varPayload: Var) => {
    return varPayload.type !== VarType.arrayFile
  }, [])

  const {
    availableVars,
    availableNodesWithParent,
  } = useAvailableVarList(id, {
    onlyLeafNodeVar: false,
    filterVar,
  })

  const filterNumberVar = useCallback((varPayload: Var) => {
    return varPayload.type === VarType.number
  }, [])

  // check_conditions and output
  const { getIterationNodeChildren, getBeforeNodesInSameBranch } = useWorkflow()
  const beforeNodes = getBeforeNodesInSameBranch(id)
  const collectChildrenNodes = getIterationNodeChildren(id)
  const canChooseVarNodes = [...beforeNodes, ...collectChildrenNodes]
  const childrenNodeVars = toNodeOutputVars(collectChildrenNodes, isChatMode)

  const handleAddCheckCondition = useCallback<HandleAddCondition>((caseId, valueSelector, varItem) => {
    const newInputs = produce(inputs, (draft) => {
      if (!draft?.check_conditions) {
        draft.check_conditions = []
      }
      draft.check_conditions.push({
          id: uuid4(),
          varType: varItem.type,
          variable_selector: valueSelector,
          comparison_operator: getOperators(varItem.type)[0],
          value: '',
        })
    })
    setInputs(newInputs)
  }, [inputs, setInputs])

  const handleRemoveCondition = useCallback<HandleRemoveCondition>((caseId, conditionId) => {
    const newInputs = produce(inputs, (draft) => {
      draft.check_conditions = draft.check_conditions.filter(item => item.id !== conditionId)
    })
    setInputs(newInputs)
  }, [inputs, setInputs])

  const handleUpdateCondition = useCallback<HandleUpdateCondition>((caseId, conditionId, newCondition) => {
    const newInputs = produce(inputs, (draft) => {
      const targetCondition = draft.check_conditions.find(item => item.id === conditionId)
        if (targetCondition)
          Object.assign(targetCondition, newCondition)
    })
    setInputs(newInputs)
  }, [inputs, setInputs])

  const handleMaxRunsChange = useCallback((size: number | string) => {
    const newInputs = produce(inputs, (draft) => {
      let limitedSize = parseInt(size as string, 10)
      if (isNaN(limitedSize))
        limitedSize = 1

      if (limitedSize < 1)
        limitedSize = 1

      if (limitedSize > 10)
        limitedSize = 10

      draft.max_runs = limitedSize as number
    })
    setInputs(newInputs)
  }, [inputs, setInputs])

  const handleOutputVarChange = useCallback((output: ValueSelector | string, _varKindType: VarKindType, varInfo?: Var) => {
    const newInputs = produce(inputs, (draft) => {
      draft.output_selector = output as ValueSelector || []
      const outputItemType = varInfo?.type || VarType.string
      draft.output_type = outputItemType
    })
    setInputs(newInputs)
  }, [inputs, setInputs])


  const handleUpdateConditionLogicalOperator = useCallback<HandleUpdateConditionLogicalOperator>((caseId, value) => {
    const newInputs = produce(inputs, (draft) => {
      draft.logical_operator = value
    })
    setInputs(newInputs)
  }, [inputs, setInputs])

  ///////////// ORIGINAL CONTENT BELOW ////////


  // single run
  // TODO: like Code Node inputs
  const iteratorInputKey = `${id}.input_selector`
  const {
    isShowSingleRun,
    showSingleRun,
    hideSingleRun,
    toVarInputs,
    runningStatus,
    handleRun: doHandleRun,
    handleStop,
    runInputData,
    setRunInputData,
    runResult,
    iterationRunResult,
  } = useOneStepRun<CollectNodeType>({
    id,
    data: inputs,
    iteratorInputKey,
    defaultRunInputData: {
      [iteratorInputKey]: [''],
    },
  })

  const [isShowCollectDetail, {
    setTrue: doShowCollectDetail,
    setFalse: doHideCollectDetail,
  }] = useBoolean(false)

  const hideCollectDetail = useCallback(() => {
    hideSingleRun()
    doHideCollectDetail()
  }, [doHideCollectDetail, hideSingleRun])

  const showCollectDetail = useCallback(() => {
    doShowCollectDetail()
  }, [doShowCollectDetail])

  const backToSingleRun = useCallback(() => {
    hideCollectDetail()
    showSingleRun()
  }, [hideCollectDetail, showSingleRun])

  const { usedOutVars, allVarObject } = (() => {
    const vars: ValueSelector[] = []
    const varObjs: Record<string, boolean> = {}
    const allVarObject: Record<string, {
      inSingleRunPassedKey: string
    }> = {}
    collectChildrenNodes.forEach((node) => {
      const nodeVars = getNodeUsedVars(node).filter(item => item && item.length > 0)
      nodeVars.forEach((varSelector) => {
        if (varSelector[0] === id) { // skip iteration node itself variable: item, index
          return
        }
        const isInIteration = isNodeInIteration(varSelector[0])
        if (isInIteration) // not pass iteration inner variable
          return

        const varSectorStr = varSelector.join('.')
        if (!varObjs[varSectorStr]) {
          varObjs[varSectorStr] = true
          vars.push(varSelector)
        }
        let passToServerKeys = getNodeUsedVarPassToServerKey(node, varSelector)
        if (typeof passToServerKeys === 'string')
          passToServerKeys = [passToServerKeys]

        passToServerKeys.forEach((key: string, index: number) => {
          allVarObject[[varSectorStr, node.id, index].join(DELIMITER)] = {
            inSingleRunPassedKey: key,
          }
        })
      })
    })
    const res = toVarInputs(vars.map((item) => {
      const varInfo = getNodeInfoById(canChooseVarNodes, item[0])
      return {
        label: {
          nodeType: varInfo?.data.type,
          nodeName: varInfo?.data.title || canChooseVarNodes[0]?.data.title, // default start node title
          variable: isSystemVar(item) ? item.join('.') : item[item.length - 1],
        },
        variable: `${item.join('.')}`,
        value_selector: item,
      }
    }))
    return {
      usedOutVars: res,
      allVarObject,
    }
  })()

  const handleRun = useCallback((data: Record<string, any>) => {
    const formattedData: Record<string, any> = {}
    Object.keys(allVarObject).forEach((key) => {
      const [varSectorStr, nodeId] = key.split(DELIMITER)
      formattedData[`${nodeId}.${allVarObject[key].inSingleRunPassedKey}`] = data[varSectorStr]
    })
    formattedData[iteratorInputKey] = data[iteratorInputKey]
    doHandleRun(formattedData)
  }, [allVarObject, doHandleRun, iteratorInputKey])

  const inputVarValues = (() => {
    const vars: Record<string, any> = {}
    Object.keys(runInputData)
      .filter(key => ![iteratorInputKey].includes(key))
      .forEach((key) => {
        vars[key] = runInputData[key]
      })
    return vars
  })()

  const setInputVarValues = useCallback((newPayload: Record<string, any>) => {
    const newVars = {
      ...newPayload,
      [iteratorInputKey]: runInputData[iteratorInputKey],
    }
    setRunInputData(newVars)
  }, [iteratorInputKey, runInputData, setRunInputData])

  const iterator = runInputData[iteratorInputKey]
  const setIterator = useCallback((newIterator: string[]) => {
    setRunInputData({
      ...runInputData,
      [iteratorInputKey]: newIterator,
    })
  }, [iteratorInputKey, runInputData, setRunInputData])

  return {
    readOnly,
    inputs,
    filterNumberVar,
    nodesOutputVars: availableVars,
    availableNodes: availableNodesWithParent,
    handleAddCheckCondition,
    handleRemoveCondition,
    handleUpdateCondition,
    collectChildrenNodes,
    childrenNodeVars,
    handleOutputVarChange,
    handleMaxRunsChange,
    handleUpdateConditionLogicalOperator,

    // for debug running
    isShowSingleRun,
    showSingleRun,
    hideSingleRun,
    isShowCollectDetail,
    showCollectDetail,
    hideCollectDetail,
    backToSingleRun,
    runningStatus,
    handleRun,
    handleStop,
    runResult,
    inputVarValues,
    setInputVarValues,
    usedOutVars,
    iterationRunResult,
  }
}

export default useConfig

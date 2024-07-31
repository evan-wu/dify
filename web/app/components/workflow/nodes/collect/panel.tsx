import type {FC} from 'react'
import React from 'react'
import {useTranslation} from 'react-i18next'
import {RiArrowRightSLine,} from '@remixicon/react'
import VarReferencePicker from '../_base/components/variable/var-reference-picker'
import Split from '../_base/components/split'
import ResultPanel from '../../run/result-panel'
import type {CollectNodeType,} from './types'
import useConfig from './use-config'
import {type NodePanelProps} from '@/app/components/workflow/types'
import Field from '@/app/components/workflow/nodes/_base/components/field'
import BeforeRunForm from '@/app/components/workflow/nodes/_base/components/before-run-form'
import ConditionAdd from "@/app/components/workflow/nodes/if-else/components/condition-add";
import {useGetAvailableVars} from "@/app/components/workflow/nodes/variable-assigner/hooks";
import ConditionList from "@/app/components/workflow/nodes/if-else/components/condition-list";
import {CaseItem, LogicalOperator} from "@/app/components/workflow/nodes/if-else/types";

const i18nPrefix = 'workflow.nodes.collect'

const Panel: FC<NodePanelProps<CollectNodeType>> = ({
  id,
  data,
}) => {
  const { t } = useTranslation()
  const getAvailableVars = useGetAvailableVars()

  const {
    readOnly,
    inputs,
    filterNumberVar,
    nodesOutputVars,
    availableNodes,
    handleAddCheckCondition,
    handleRemoveCondition,
    handleUpdateCondition,
    collectChildrenNodes,
    handleMaxRunsChange,
    handleUpdateConditionLogicalOperator,
    childrenNodeVars,
    handleOutputVarChange,
  } = useConfig(id, data)

  const caseItem = {case_id: 'true', id: 'true',
    conditions: data.check_conditions, logical_operator: data.logical_operator || LogicalOperator.and} as CaseItem

  return (
    <div className='mt-2'>
      <div className='px-4 pb-4 space-y-4'>
        <Field
          title='设置退出条件'
          >
          <div className='flex items-center justify-between pl-[60px] pr-[30px] mt-1'>
            <ConditionAdd
                    disabled={readOnly}
                    caseId={`collect-check-${id}`}
                    variables={childrenNodeVars}
                    onSelectVariable={handleAddCheckCondition}
            />
          </div>
        </Field>
      </div>

      { !!inputs.check_conditions?.length && (
        <div className='mb-2'>
          <ConditionList
            disabled={readOnly}
            caseItem={caseItem}
            onUpdateCondition={handleUpdateCondition}
            onRemoveCondition={handleRemoveCondition}
            onUpdateConditionLogicalOperator={handleUpdateConditionLogicalOperator}
            nodesOutputVars={nodesOutputVars}
            availableNodes={availableNodes}
            numberVariables={getAvailableVars(id, '', filterNumberVar)}
          />
        </div>
        )
      }

      <Split />
      <div className='mt-2 px-4 pb-4 space-y-4'>
        <Field
          title='最大采集次数'
          operations={(
            <div className='flex items-center h-[18px] px-1 border border-black/8 rounded-[5px] text-xs font-medium text-gray-500 capitalize'>Number</div>
          )}
        >
          <input
            value={(inputs?.max_runs || 1) as number}
            className='shrink-0 block ml-4 pl-3 w-12 h-8 appearance-none outline-none rounded-lg bg-gray-100 text-[13px] text-gra-900'
            type='number'
            min={1}
            max={10}
            step={1}
            onChange={e => handleMaxRunsChange(e.target.value)}
            // onBlur={handleBlur}
            disabled={readOnly}
          />
        </Field>
      </div>
      <Split />
      <div className='mt-2 px-4 pb-4 space-y-4'>
        <Field
          title={t(`${i18nPrefix}.output`)}
          operations={(
            <div className='flex items-center h-[18px] px-1 border border-black/8 rounded-[5px] text-xs font-medium text-gray-500 capitalize'>Any</div>
          )}
        >
          <VarReferencePicker
            readonly={readOnly}
            nodeId={id}
            isShowNodeName
            value={inputs.output_selector || []}
            onChange={handleOutputVarChange}
            availableNodes={collectChildrenNodes}
            availableVars={childrenNodeVars}
          />
        </Field>
      </div>
    </div>
  )
}

export default React.memo(Panel)

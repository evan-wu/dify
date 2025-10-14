import { BlockEnum } from '../../types'
import type { NodeDefault } from '../../types'
import type { CollectNodeType } from './types'
import { genNodeMetaData } from '@/app/components/workflow/utils'
import { LogicalOperator } from '@/app/components/workflow/nodes/if-else/types'
import { BlockClassificationEnum } from '@/app/components/workflow/block-selector/types'
const i18nPrefix = 'workflow'

const metaData = genNodeMetaData({
  classification: BlockClassificationEnum.Logic,
  sort: 2,
  type: BlockEnum.Collect,
  isTypeFixed: true,
})
const nodeDefault: NodeDefault<CollectNodeType> = {
  metaData,
  defaultValue: {
    start_node_id: '',
    max_runs: 1,
    check_conditions: [],
    logical_operator: LogicalOperator.and,
    output_selector: [],
    _children: [],
  },
  checkValid(payload: CollectNodeType, t: any) {
    let errorMessages = ''

    if (!errorMessages && (!payload.check_conditions || payload.check_conditions.length === 0))
      errorMessages = t(`${i18nPrefix}.errorMsg.fieldRequired`, { field: '需要设置采集退出条件' })

    if (!errorMessages && (!payload.output_selector || payload.output_selector.length === 0))
      errorMessages = t(`${i18nPrefix}.errorMsg.fieldRequired`, { field: t(`${i18nPrefix}.nodes.collect.output`) })

    return {
      isValid: !errorMessages,
      errorMessage: errorMessages,
    }
  },
}

export default nodeDefault

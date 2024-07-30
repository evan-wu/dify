import type {
  BlockEnum,
  CommonNodeType,
  ValueSelector,
  VarType,
} from '@/app/components/workflow/types'
import {Condition, LogicalOperator} from "@/app/components/workflow/nodes/if-else/types";

export type CollectNodeType = CommonNodeType & {
  startNodeType?: BlockEnum
  start_node_id: string // start node id in the iteration
  collect_id?: string
  max_runs: number
  check_conditions: Condition[]
  logical_operator: LogicalOperator
  output_selector: ValueSelector
  output_type: VarType // output type.
}

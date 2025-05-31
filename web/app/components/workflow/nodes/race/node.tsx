import type { FC } from 'react'
import {
  memo,
  useMemo,
} from 'react'
import type { NodeProps } from 'reactflow'
import { useTranslation } from 'react-i18next'
import { BlockEnum } from '../../types'
import type { RaceNodeType } from './types'
import { RaceStrategy } from './types'
import { NodeSourceHandle, NodeTargetHandle } from '../_base/components/node-handle'

const i18nPrefix = 'workflow.nodes.race'

const Node: FC<NodeProps<RaceNodeType>> = (props) => {
  const { t } = useTranslation()
  const { id, data } = props

  const { race_strategy, variables, timeout_seconds, max_winners } = data

  const strategyLabel = useMemo(() => {
    const strategyMap = {
      [RaceStrategy.FIRST_COMPLETE]: t(`${i18nPrefix}.strategy.firstComplete`),
      [RaceStrategy.FASTEST_VALID]: t(`${i18nPrefix}.strategy.fastestValid`),
      [RaceStrategy.TIMEOUT_BEST]: t(`${i18nPrefix}.strategy.timeoutBest`),
      [RaceStrategy.QUALITY_RACE]: t(`${i18nPrefix}.strategy.qualityRace`),
    }
    return strategyMap[race_strategy] || race_strategy
  }, [race_strategy, t])

  const raceInfo = useMemo(() => {
    const info = []
    if (variables?.length) {
      info.push(`${variables.length} ${t(`${i18nPrefix}.competitors`)}`)
    }
    if (timeout_seconds) {
      info.push(`${timeout_seconds}s ${t(`${i18nPrefix}.timeout.title`)}`)
    }
    if (max_winners > 1) {
      info.push(`${max_winners} ${t(`${i18nPrefix}.winners`)}`)
    }
    return info
  }, [variables, timeout_seconds, max_winners, t])

  return (
    <>
      <div className="flex flex-col">
        <div className="px-3 py-2">
          <div className="text-xs text-gray-500 mb-2">
            {t(`${i18nPrefix}.strategy.title`)}: {strategyLabel}
          </div>
          
          {raceInfo.length > 0 && (
            <div className="text-xs text-gray-400 space-y-1">
              {raceInfo.map((info, index) => (
                <div key={index} className="flex items-center">
                  <div className="w-1 h-1 bg-gray-400 rounded-full mr-2" />
                  {info}
                </div>
              ))}
            </div>
          )}
          
          {(!variables || variables.length === 0) && (
            <div className="text-xs text-red-400">
              {t(`${i18nPrefix}.noVariables`)}
            </div>
          )}
        </div>
      </div>
    </>
  )
}

export default memo(Node) 

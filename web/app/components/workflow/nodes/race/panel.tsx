import type { FC } from 'react'
import React, { useCallback } from 'react'
import { useTranslation } from 'react-i18next'
import { produce } from 'immer'
import type { RaceNodeType } from './types'
import { RaceStrategy, WinCondition } from './types'
import type { NodePanelProps, ValueSelector } from '@/app/components/workflow/types'
import { BlockEnum } from '@/app/components/workflow/types'
import useConfig from './use-config'
import Field from '@/app/components/workflow/nodes/_base/components/field'
import Split from '@/app/components/workflow/nodes/_base/components/split'
import OutputVars, { VarItem } from '@/app/components/workflow/nodes/_base/components/output-vars'
import AddButton from '@/app/components/workflow/nodes/_base/components/add-button'
import { RiDeleteBinLine } from '@remixicon/react'
import VarReferencePicker from '@/app/components/workflow/nodes/_base/components/variable/var-reference-picker'

const i18nPrefix = 'workflow.nodes.race'

const Panel: FC<NodePanelProps<RaceNodeType>> = ({
  id,
  data,
}) => {
  const { t } = useTranslation()

  const {
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
  } = useConfig(id, data)

  return (
    <div className="mt-2">
      {/* Race Strategy */}
      <div className='px-4 pb-4 space-y-4'>
        <Field
          title={t(`${i18nPrefix}.strategy.title`)}
        >
          <select
            className="w-full px-3 py-2 text-sm bg-white border border-gray-200 rounded-lg focus:outline-none focus:ring-1 focus:ring-blue-500"
            value={data.race_strategy}
            onChange={handleRaceStrategyChange}
            disabled={readOnly}
          >
            <option value={RaceStrategy.FIRST_COMPLETE}>
              {t(`${i18nPrefix}.strategy.firstComplete`)}
            </option>
            <option value={RaceStrategy.FASTEST_VALID}>
              {t(`${i18nPrefix}.strategy.fastestValid`)}
            </option>
            <option value={RaceStrategy.TIMEOUT_BEST}>
              {t(`${i18nPrefix}.strategy.timeoutBest`)}
            </option>
            <option value={RaceStrategy.QUALITY_RACE}>
              {t(`${i18nPrefix}.strategy.qualityRace`)}
            </option>
          </select>
        </Field>
      </div>
      <Split />

      {/* Win Condition */}
      <div className='mt-2 px-4 pb-4 space-y-4'>
        <Field
          title={t(`${i18nPrefix}.winCondition.title`)}
        >
          <select
            className="w-full px-3 py-2 text-sm bg-white border border-gray-200 rounded-lg focus:outline-none focus:ring-1 focus:ring-blue-500"
            value={data.win_condition}
            onChange={handleWinConditionChange}
            disabled={readOnly}
          >
            <option value={WinCondition.ANY_RESULT}>
              {t(`${i18nPrefix}.winCondition.anyResult`)}
            </option>
            <option value={WinCondition.NO_ERROR}>
              {t(`${i18nPrefix}.winCondition.noError`)}
            </option>
            <option value={WinCondition.CUSTOM_VALIDATION}>
              {t(`${i18nPrefix}.winCondition.customValidation`)}
            </option>
          </select>
        </Field>
      </div>

      {/* Custom Validation Expression (only for custom validation) */}
      {data.win_condition === WinCondition.CUSTOM_VALIDATION && (
        <div className='mt-2 px-4 pb-4 space-y-4'>
          <Field
            title={t(`${i18nPrefix}.validationExpression.title`)}
          >
            <>
              <textarea
                className="w-full px-3 py-2 text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-1 focus:ring-blue-500"
                value={data.validation_expression || ''}
                onChange={handleValidationExpressionChange}
                disabled={readOnly}
                rows={3}
                placeholder={t(`${i18nPrefix}.validationExpression.placeholder`)}
              />
              <div className="text-xs text-gray-500 mt-1">
                {t(`${i18nPrefix}.validationExpression.description`)}
              </div>
            </>
          </Field>
        </div>
      )}

      {/* Scoring Expression (only for quality race) */}
      {data.race_strategy === RaceStrategy.QUALITY_RACE && (
        <div className='mt-2 px-4 pb-4 space-y-4'>
          <Field
            title={t(`${i18nPrefix}.scoringExpression.title`)}
          >
            <>
              <textarea
                className="w-full px-3 py-2 text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-1 focus:ring-blue-500"
                value={data.scoring_expression || ''}
                onChange={handleScoringExpressionChange}
                disabled={readOnly}
                rows={3}
                placeholder={t(`${i18nPrefix}.scoringExpression.placeholder`)}
              />
              <div className="text-xs text-gray-500 mt-1">
                {t(`${i18nPrefix}.scoringExpression.description`)}
              </div>
            </>
          </Field>
        </div>
      )}

      <div className='mt-2 px-4 pb-4 space-y-4'>
        {/* Timeout */}
        <Field
          title={t(`${i18nPrefix}.timeout.title`)}
        >
          <>
            <input
              type="number"
              className="w-full px-3 py-2 text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-1 focus:ring-blue-500"
              value={data.timeout_seconds}
              onChange={handleTimeoutChange}
              disabled={readOnly}
              min="1"
              step="1"
            />
            <div className="text-xs text-gray-500 mt-1">
              {t(`${i18nPrefix}.timeout.description`)}
            </div>
          </>
        </Field>
      </div>

      <div className='mt-2 px-4 pb-4 space-y-4'>
        {/* Max Winners */}
        <Field
          title={t(`${i18nPrefix}.maxWinners.title`)}
        >
          <>
            <input
              type="number"
              className="w-full px-3 py-2 text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-1 focus:ring-blue-500"
              value={data.max_winners}
              onChange={handleMaxWinnersChange}
              disabled={readOnly}
              min="1"
              step="1"
            />
            <div className="text-xs text-gray-500 mt-1">
              {t(`${i18nPrefix}.maxWinners.description`)}
            </div>
          </>
        </Field>
      </div>

      <div className='mt-2 px-4 pb-4 space-y-4'>
        {/* Variables to Race */}
        <Field
          title={t(`${i18nPrefix}.variables.title`)}
        >
          <div className="space-y-2">
            {data.variables?.map((variable, index) => (
              <div key={index} className="flex items-center space-x-2">
                <div className="flex-1">
                  <VarReferencePicker
                    readonly={readOnly}
                    nodeId={id}
                    isShowNodeName
                    value={variable}
                    onChange={(value) => handleVariableChange(index, value as ValueSelector)}
                    filterVar={filterVar}
                  />
                </div>
                {!readOnly && (
                  <button
                    type="button"
                    onClick={() => handleVariableRemove(index)}
                    className="p-1 text-gray-400 hover:text-red-500 transition-colors"
                  >
                    <RiDeleteBinLine className="w-4 h-4" />
                  </button>
                )}
              </div>
            ))}

            {!readOnly && (
              <AddButton
                onClick={handleVariableAdd}
                text={t(`${i18nPrefix}.variables.add`)}
              />
            )}
          </div>
        </Field>
      </div>

      {/* Advanced Options */}
      <Split />

      <div className='mt-2 px-4 pb-4 space-y-4'>
        <Field
          title={t(`${i18nPrefix}.advanced.title`)}
        >
          <div className="space-y-3">
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={data.fail_on_timeout}
                onChange={handleFailOnTimeoutChange}
                disabled={readOnly}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <span className="text-sm text-gray-700">
                {t(`${i18nPrefix}.advanced.failOnTimeout`)}
              </span>
            </label>

            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={data.fail_on_all_errors}
                onChange={handleFailOnAllErrorsChange}
                disabled={readOnly}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <span className="text-sm text-gray-700">
                {t(`${i18nPrefix}.advanced.failOnAllErrors`)}
              </span>
            </label>
          </div>
        </Field>
      </div>

      {/* Output Variables */}
      <Split />
      <OutputVars>
        <>
          <VarItem
            name="race_winner"
            type="object"
            description={t(`${i18nPrefix}.output.raceWinner`)}
          />
          <VarItem
            name="race_status"
            type="string"
            description={t(`${i18nPrefix}.output.raceStatus`)}
          />
          <VarItem
            name="total_competitors"
            type="number"
            description={t(`${i18nPrefix}.output.totalCompetitors`)}
          />
          <VarItem
            name="total_winners"
            type="number"
            description={t(`${i18nPrefix}.output.totalWinners`)}
          />
        </>
      </OutputVars>
    </div>
  )
}

export default React.memo(Panel) 

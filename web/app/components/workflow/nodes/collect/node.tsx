import type { FC } from 'react'
import {
  memo,
  useEffect,
} from 'react'
import {
  Background,
  useNodesInitialized,
  useViewport,
} from 'reactflow'
import { CollectStartNodeDumb } from '../collect-start'
import { useNodeCollectInteractions } from './use-interactions'
import type { CollectNodeType } from './types'
import AddBlock from './add-block'
import cn from '@/utils/classnames'
import type { NodeProps } from '@/app/components/workflow/types'

const Node: FC<NodeProps<CollectNodeType>> = ({
  id,
  data,
}) => {
  const { zoom } = useViewport()
  const nodesInitialized = useNodesInitialized()
  const { handleNodeCollectRerender } = useNodeCollectInteractions()

  useEffect(() => {
    if (nodesInitialized)
      handleNodeCollectRerender(id)
  }, [nodesInitialized, id, handleNodeCollectRerender])

  return (
    <div className={cn(
      'relative h-full min-h-[90px] w-full min-w-[240px] rounded-2xl bg-workflow-canvas-workflow-bg',
    )}>
      <Background
        id={`collect-background-${id}`}
        className='!z-0 rounded-2xl'
        gap={[14 / zoom, 14 / zoom]}
        size={2 / zoom}
        color='var(--color-workflow-canvas-workflow-dot-color)'
      />
      {
        data._isCandidate && (
          <CollectStartNodeDumb />
        )
      }
      {
        data._children!.length === 1 && (
          <AddBlock
            collectNodeId={id}
            collectNodeData={data}
          />
        )
      }
    </div>
  )
}

export default memo(Node)

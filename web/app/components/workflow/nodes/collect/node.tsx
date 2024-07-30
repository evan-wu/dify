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
      'relative min-w-[258px] min-h-[118px] w-full h-full rounded-2xl bg-[#F0F2F7]/90',
    )}>
      <Background
        id={`collect-background-${id}`}
        className='rounded-2xl !z-0'
        gap={[14 / zoom, 14 / zoom]}
        size={2 / zoom}
        color='#E4E5E7'
      />
      <AddBlock
        collectNodeId={id}
        collectNodeData={data}
      />
    </div>
  )
}

export default memo(Node)

import React from 'react'
import { FC } from 'react'
import { GroupConfig } from 'konva/lib/Group'
import { Group, Line } from 'react-konva'
import { isCanvasMaskLine } from '../atoms/canvasTypes'
import { useRecoilValue } from 'recoil'
import { layerStateAtom } from '../atoms/canvas.atoms'

// import {canvasSelector} from 'canvas/store/canvasSelectors';
// import _ from 'lodash';

// export const canvasLinesSelector = createSelector(
// 	[canvasSelector],
// 	(canvas) => {
// 		return {objects: canvas.layerState.objects};
// 	},
// 	{
// 		memoizeOptions: {
// 			resultEqualityCheck: _.isEqual,
// 		},
// 	}
// );

type InpaintingCanvasLinesProps = GroupConfig

/**
 * Draws the lines which comprise the mask.
 *
 * Uses globalCompositeOperation to handle the brush and eraser tools.
 */
export const CanvasMaskLines: FC<InpaintingCanvasLinesProps> = (props) => {
  const { ...rest } = props
  const layerState = useRecoilValue(layerStateAtom)
  const { objects } = layerState
  // const { objects } = useAppSelector(canvasLinesSelector);

  return (
    <Group listening={false} {...rest}>
      {objects.filter(isCanvasMaskLine).map((line, i) => (
        <Line
          key={i}
          points={line.points}
          stroke={'rgb(0,0,0)'} // The lines can be any color, just need alpha > 0
          strokeWidth={line.strokeWidth * 2}
          tension={0}
          lineCap="round"
          lineJoin="round"
          shadowForStrokeEnabled={false}
          listening={false}
          globalCompositeOperation={line.tool === 'brush' ? 'source-over' : 'destination-out'}
        />
      ))}
    </Group>
  )
}

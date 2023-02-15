import React from 'react'
import { FC, useLayoutEffect } from 'react'
import _ from 'lodash'
import { useRecoilState } from 'recoil'
import { ImageUploader } from './components'
import {
  CanvasResizer,
  Canvas,
  CanvasOutpaintingControls
} from './canvas.components'
import { doesCanvasNeedScalingAtom } from './atoms/canvas.atoms'

// import { canvasSelector } from 'canvas/store/canvasSelectors';
// import { setDoesCanvasNeedScaling } from 'canvas/store/canvasSlice';

// const selector = createSelector(
//   [canvasSelector],
//   (canvas) => {
//     const { doesCanvasNeedScaling } = canvas;
//     return {
//       doesCanvasNeedScaling,
//     };
//   },
//   {
//     memoizeOptions: {
//       resultEqualityCheck: _.isEqual,
//     },
//   }
// );

export const UnifiedCanvas: FC = () => {
  // const { doesCanvasNeedScaling } = useAppSelector(selector);

  const [doesCanvasNeedScaling, setDoesCanvasNeedScaling] = useRecoilState(
    doesCanvasNeedScalingAtom
  )

  useLayoutEffect(() => {
    setDoesCanvasNeedScaling(true)

    const resizeCallback = _.debounce(() => {
      setDoesCanvasNeedScaling(true)
    }, 250)

    window.addEventListener('resize', resizeCallback)

    return () => window.removeEventListener('resize', resizeCallback)
  }, [])

  return (
    <div>
      <ImageUploader>
        <div className="app-content">
          <div className="workarea-wrapper inpainting-workarea-overrides">
            <div className="workarea-main">
              <div
                className="workarea-children-wrapper"
                // onDrop={handleDrop}
              >
                <div className="workarea-split-view-left">
                  <div className="inpainting-main-area">
                    <CanvasOutpaintingControls />
                    <div className="inpainting-canvas-area">
                      {doesCanvasNeedScaling ? <CanvasResizer /> : <Canvas />}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </ImageUploader>
    </div>
  )
}

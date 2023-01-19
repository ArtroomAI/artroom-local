import { useHotkeys } from 'react-hotkeys-hook';
import { useRef } from 'react';
import { useRecoilState, useRecoilValue, useSetRecoilState } from 'recoil';
import { CanvasTool } from '../atoms/canvasTypes';
import { getCanvasStage } from '../util';
import {
  toolSelector,
  clearMaskAction,
  setIsMaskEnabledAction,
  shouldShowBoundingBoxAtom,
  resetCanvasInteractionStateAction,
  shouldSnapToGridAtom,
  isMaskEnabledAtom,
  isStagingSelector,
} from '../atoms/canvas.atoms';

// import _ from 'lodash';
// import { activeTabNameSelector } from 'options/store/optionsSelectors';
// import {
// 	clearMask,
// 	resetCanvasInteractionState,
// 	setIsMaskEnabled,
// 	setShouldShowBoundingBox,
// 	setShouldSnapToGrid,
// 	setTool,
// } from 'canvas/store/canvasSlice';
// import {
// 	canvasSelector,
// 	isStagingSelector,
// } from 'canvas/store/canvasSelectors';

// const selector = createSelector(
// 	[canvasSelector, activeTabNameSelector, isStagingSelector],
// 	(canvas, activeTabName, isStaging) => {
// 		const {
// 			cursorPosition,
// 			shouldLockBoundingBox,
// 			shouldShowBoundingBox,
// 			tool,
// 			isMaskEnabled,
// 			shouldSnapToGrid,
// 		} = canvas;

// 		return {
// 			activeTabName,
// 			isCursorOnCanvas: Boolean(cursorPosition),
// 			shouldLockBoundingBox,
// 			shouldShowBoundingBox,
// 			tool,
// 			isStaging,
// 			isMaskEnabled,
// 			shouldSnapToGrid,
// 		};
// 	},
// 	{
// 		memoizeOptions: {
// 			resultEqualityCheck: _.isEqual,
// 		},
// 	},
// );

export const useInpaintingCanvasHotkeys = () => {
  // const {
  // 	activeTabName,
  // 	shouldShowBoundingBox,
  // 	tool,
  // 	isStaging,
  // 	isMaskEnabled,
  // 	shouldSnapToGrid,
  // } = useAppSelector(selector);

  const [tool, setTool] = useRecoilState(toolSelector);
  const clearMask = useSetRecoilState(clearMaskAction);
  const setIsMaskEnabled = useSetRecoilState(setIsMaskEnabledAction);
  const resetCanvasInteractionState = useSetRecoilState(
    resetCanvasInteractionStateAction
  );
  const [shouldSnapToGrid, setShouldSnapToGrid] =
    useRecoilState(shouldSnapToGridAtom);
  const [shouldShowBoundingBox, setShouldShowBoundingBox] = useRecoilState(
    shouldShowBoundingBoxAtom
  );
  const isMaskEnabled = useRecoilValue(isMaskEnabledAtom);
  const isStaging = useRecoilValue(isStagingSelector);

  const previousToolRef = useRef<CanvasTool | null>(null);

  const canvasStage = getCanvasStage();

  // Beta Keys
  const handleClearMask = () => clearMask();

  useHotkeys(
    ['shift+c'],
    () => {
      handleClearMask();
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    []
  );

  const handleToggleEnableMask = () => setIsMaskEnabled(!isMaskEnabled);

  useHotkeys(
    ['h'],
    () => {
      handleToggleEnableMask();
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [isMaskEnabled]
  );

  useHotkeys(
    ['n'],
    () => {
      setShouldSnapToGrid(!shouldSnapToGrid);
    },
    {
      enabled: true,
      preventDefault: true,
    },
    [shouldSnapToGrid]
  );
  //

  useHotkeys(
    'esc',
    () => {
      resetCanvasInteractionState();
    },
    {
      enabled: () => true,
      preventDefault: true,
    }
  );

  useHotkeys(
    'shift+h',
    () => {
      setShouldShowBoundingBox(!shouldShowBoundingBox);
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [shouldShowBoundingBox]
  );

  useHotkeys(
    ['space'],
    (e: KeyboardEvent) => {
      e.preventDefault();
      if (e.repeat) return;

      canvasStage?.container().focus();

      if (tool !== 'move') {
        previousToolRef.current = tool;
        setTool('move');
      }

      if (
        tool === 'move' &&
        previousToolRef.current &&
        previousToolRef.current !== 'move'
      ) {
        setTool(previousToolRef.current);
        previousToolRef.current = 'move';
      }
    },
    {
      keyup: true,
      keydown: true,
      preventDefault: true,
    },
    [tool, previousToolRef]
  );
};

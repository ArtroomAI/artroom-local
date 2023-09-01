import React from 'react'
import { FC, ChangeEvent } from 'react'
import { Flex } from '@chakra-ui/react'
import { FaWrench } from 'react-icons/fa'
import { useHotkeys } from 'react-hotkeys-hook'
import { useRecoilState } from 'recoil'
import { Popover, Checkbox, IconButton } from '../../components'
import { EmptyTempFolderButtonModal } from '../ClearTempFolderButtonModal'
import { ClearCanvasHistoryButtonModal } from '../ClearCanvasHistoryButtonModal'
import {
  shouldDarkenOutsideBoundingBoxAtom,
  shouldSnapToGridAtom,
  shouldAutoSaveAtom,
  shouldShowGridAtom,
  shouldShowCanvasDebugInfoAtom,
  shouldRestrictStrokesToBoxAtom,
  shouldCropToBoundingBoxOnSaveAtom,
  shouldShowIntermediatesAtom,
} from '../../atoms/canvas.atoms'
import { paletteFixState } from '../../../../SettingsManager'

export const CanvasSettingsButtonPopover: FC = () => {
  // const {
  // 	shouldAutoSave,
  // 	shouldCropToBoundingBoxOnSave,
  // 	shouldDarkenOutsideBoundingBox,
  // 	shouldShowCanvasDebugInfo,
  // 	shouldShowGrid,
  // 	shouldShowIntermediates,
  // 	shouldSnapToGrid,
  // 	shouldRestrictStrokesToBox,
  // } = useAppSelector(canvasControlsSelector);

  const [shouldDarkenOutsideBoundingBox, setShouldDarkenOutsideBoundingBox] = useRecoilState(
    shouldDarkenOutsideBoundingBoxAtom
  )
  const [shouldSnapToGrid, setShouldSnapToGrid] = useRecoilState(shouldSnapToGridAtom)
  const [shouldAutoSave, setShouldAutoSave] = useRecoilState(shouldAutoSaveAtom)
  const [shouldShowGrid, setShouldShowGrid] = useRecoilState(shouldShowGridAtom)
  const [shouldShowCanvasDebugInfo, setShouldShowCanvasDebugInfo] = useRecoilState(
    shouldShowCanvasDebugInfoAtom
  )
  const [shouldRestrictStrokesToBox, setShouldRestrictStrokesToBox] = useRecoilState(
    shouldRestrictStrokesToBoxAtom
  )
  const [shouldCropToBoundingBoxOnSave, setShouldCropToBoundingBoxOnSave] = useRecoilState(
    shouldCropToBoundingBoxOnSaveAtom
  )
  const [shouldShowIntermediates, setShouldShowIntermediates] = useRecoilState(
    shouldShowIntermediatesAtom
  )
  const [paletteFix, setPaletteFix] = useRecoilState(paletteFixState)

  useHotkeys(
    ['n'],
    () => {
      setShouldSnapToGrid(!shouldSnapToGrid)
    },
    {
      enabled: true,
      preventDefault: true,
    },
    [shouldSnapToGrid]
  )

  const handleChangeShouldSnapToGrid = (e: ChangeEvent<HTMLInputElement>) =>
    setShouldSnapToGrid(e.target.checked)

  return (
    <Popover
      trigger="hover"
      triggerComponent={
        <IconButton tooltip="Canvas Settings" aria-label="Canvas Settings" icon={<FaWrench />} />
      }
    >
      <Flex direction="column" gap="0.5rem">
        <Checkbox
          isDisabled={true}
          label="Show Intermediates (Coming Soon)"
          isChecked={shouldShowIntermediates}
          onChange={(e) => setShouldShowIntermediates(e.target.checked)}
        />
        <Checkbox
          label="Show Grid"
          isChecked={shouldShowGrid}
          onChange={(e) => setShouldShowGrid(e.target.checked)}
        />
        <Checkbox
          label="Snap to Grid"
          isChecked={shouldSnapToGrid}
          onChange={handleChangeShouldSnapToGrid}
        />
        <Checkbox
          label="Darken Outside Selection"
          isChecked={shouldDarkenOutsideBoundingBox}
          onChange={(e) => setShouldDarkenOutsideBoundingBox(e.target.checked)}
        />
        <Checkbox
          label="Save Box Region Only"
          isChecked={shouldCropToBoundingBoxOnSave}
          onChange={(e) => setShouldCropToBoundingBoxOnSave(e.target.checked)}
        />
        <Checkbox
          label="Limit Strokes to Box"
          isChecked={shouldRestrictStrokesToBox}
          onChange={(e) => setShouldRestrictStrokesToBox(e.target.checked)}
        />
        <Checkbox
          label="Use Palette Fix"
          isChecked={paletteFix}
          onChange={(e) => setPaletteFix(e.target.checked)}
        />
        <Checkbox
          label="Show Canvas Debug Info"
          isChecked={shouldShowCanvasDebugInfo}
          onChange={(e) => setShouldShowCanvasDebugInfo(e.target.checked)}
        />
        <ClearCanvasHistoryButtonModal />
        <EmptyTempFolderButtonModal />
      </Flex>
    </Popover>
  )
}

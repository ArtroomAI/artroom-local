import React from 'react'
import { FC } from 'react'
import { ButtonGroup, Flex } from '@chakra-ui/react'
import { FaMask, FaTrash } from 'react-icons/fa'
import { GiDualityMask } from 'react-icons/gi'
import { useHotkeys } from 'react-hotkeys-hook'
import { useRecoilState, useRecoilValue, useSetRecoilState } from 'recoil'
import {
  Button,
  Popover,
  Checkbox,
  ColorPicker,
  IconButton
} from '../../components'
import {
  layerAtom,
  maskColorAtom,
  shouldPreserveMaskedAreaAtom,
  clearMaskAction,
  setIsMaskEnabledAction,
  isStagingSelector
} from '../../atoms/canvas.atoms'

export const CanvasMaskOptions: FC = () => {
  const [layer, setLayer] = useRecoilState(layerAtom)
  const [maskColor, setMaskColor] = useRecoilState(maskColorAtom)
  const [shouldPreserveMaskedArea, setShouldPreserveMaskedArea] =
    useRecoilState(shouldPreserveMaskedAreaAtom)
  const clearMask = useSetRecoilState(clearMaskAction)
  const [isMaskEnabled, setIsMaskEnabled] = useRecoilState(
    setIsMaskEnabledAction
  )
  const isStaging = useRecoilValue(isStagingSelector)

  useHotkeys(
    ['q'],
    () => {
      handleToggleMaskLayer()
    },
    {
      enabled: () => !isStaging,
      preventDefault: true
    },
    [layer]
  )

  useHotkeys(
    ['shift+c'],
    () => {
      handleClearMask()
    },
    {
      enabled: () => !isStaging,
      preventDefault: true
    },
    []
  )

  useHotkeys(
    ['h'],
    () => {
      handleToggleEnableMask()
    },
    {
      enabled: () => !isStaging,
      preventDefault: true
    },
    [isMaskEnabled]
  )

  const handleToggleMaskLayer = () => {
    setLayer(layer === 'mask' ? 'base' : 'mask')
  }

  const handleClearMask = () => clearMask()

  const handleToggleEnableMask = () => setIsMaskEnabled(!isMaskEnabled)

  return (
    <Popover
      trigger="hover"
      triggerComponent={
        <ButtonGroup>
          <IconButton
            aria-label="Use Mask (H)"
            tooltip="Use Mask (H)"
            icon={<GiDualityMask />}
            style={
              layer === 'mask'
                ? { backgroundColor: 'var(--accent-color)' }
                : { backgroundColor: 'var(--btn-base-color)' }
            }
            isDisabled={isStaging}
            onClick={handleToggleEnableMask}
          />
        </ButtonGroup>
      }
    >
      <Flex direction="column" gap="0.5rem" align="center">
        <Checkbox
          label="Preserve Masked Area"
          isChecked={shouldPreserveMaskedArea}
          onChange={e => setShouldPreserveMaskedArea(e.target.checked)}
        />
        <ColorPicker
          style={{ paddingTop: '0.5rem', paddingBottom: '0.5rem' }}
          color={maskColor}
          onChange={newColor => setMaskColor(newColor)}
        />
        <Button size="sm" leftIcon={<FaTrash />} onClick={handleClearMask}>
          Clear Mask (Shift+C)
        </Button>
      </Flex>
    </Popover>
  )
}

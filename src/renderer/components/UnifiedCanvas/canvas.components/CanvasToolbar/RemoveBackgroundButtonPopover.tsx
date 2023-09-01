import React, { FC, useState } from 'react'
import { Button, Flex, useToast } from '@chakra-ui/react'
import { useRecoilValue } from 'recoil'
import { Popover, IconButton, Slider, Select } from '../../components'
import { MdOutlineFlipToBack } from 'react-icons/md'
import axios from 'axios'
import { getCanvasBaseLayer, layerToDataURL } from '../../util'
import { stageScaleAtom, stageCoordinatesAtom } from '../../atoms/canvas.atoms'

export const RemoveBackgroundButtonPopover: FC = () => {
  const toast = useToast({})
  const [upscaler, setUpscaler] = useState('RealESRGAN')
  const [upscale_factor, setUpscaleFactor] = useState(2)
  const stageScale = useRecoilValue(stageScaleAtom)
  const stageCoordinates = useRecoilValue(stageCoordinatesAtom)

  function handleUpscale() {
    toast({
      title: 'Recieved!',
      description: "You'll get a notification 🔔 when your upscale is ready!",
      status: 'success',
      position: 'top',
      duration: 1500,
      isClosable: false,
      containerStyle: {
        pointerEvents: 'none',
      },
    })
    const canvasBaseLayer = getCanvasBaseLayer()

    const { dataURL, boundingBox: originalBoundingBox } = layerToDataURL(
      canvasBaseLayer,
      stageScale,
      stageCoordinates
    )

    const output = {
      upscale_image: dataURL,
      upscaler,
      upscale_factor,
      upscale_dest: '',
    }
    axios
      .post('http://127.0.0.1:5300/upscale_canvas', output, {
        headers: { 'Content-Type': 'application/json' },
      })
      .then((result) => {
        if (result.data.status === 'Failure') {
          toast({
            title: 'Background Removal Failed',
            description: result.data.status_message,
            status: 'error',
            position: 'top',
            duration: 5000,
            isClosable: false,
            containerStyle: {
              pointerEvents: 'none',
            },
          })
        } else {
          toast({
            title: 'Background Removed Completed',
            status: 'success',
            position: 'top',
            duration: 1000,
            isClosable: false,
            containerStyle: {
              pointerEvents: 'none',
            },
          })
        }
      })
  }

  return (
    <Popover
      trigger="hover"
      triggerComponent={
        <IconButton
          tooltip="Remove Background"
          aria-label="Remove Background"
          icon={<MdOutlineFlipToBack />}
        />
      }
    >
      <Flex minWidth="15rem" direction="column" gap="1rem" width="100%">
        <Slider
          step={0.5}
          label="Size"
          value={upscale_factor}
          withInput
          max={4}
          onChange={(newSize) => setUpscaleFactor(newSize)}
          sliderNumberInputProps={{ max: 4 }}
          inputReadOnly={false}
        />
        <Select
          label="Upscaler"
          id="upscaler"
          name="upscaler"
          onChange={(event) => setUpscaler(event.target.value)}
          value={upscaler}
          variant="outline"
          w="300px"
          validValues={[
            'RealESRGAN',
            'RealESRGAN-Anime',
            'GFPGANv1.3',
            'GFPGANv1.4',
            'RestoreFormer',
          ]}
        />
        <Button onClick={handleUpscale}>Remove Background</Button>
      </Flex>
    </Popover>
  )
}

import React from 'react'
import { useRecoilState, useRecoilValue } from 'recoil'
import * as atom from '../../../atoms/atoms'
import { Box, Image, IconButton, ButtonGroup, Tooltip, Flex, useToast } from '@chakra-ui/react'
import { FaTrashAlt, FaSave } from 'react-icons/fa'
import { batchNameState, removeBackgroundState, imageSavePathState } from '../../../SettingsManager'
import path from 'path'
import fs from 'fs'

const RemoveBackgroundPreview = () => {
  const toast = useToast({})

  const [removeBackgroundPreview, setRemoveBackgroundPreview] = useRecoilState(
    atom.removeBackgroundPreviewState
  )
  const removeBackground = useRecoilValue(removeBackgroundState)

  // For remove bg
  const batchName = useRecoilValue(batchNameState)
  const imageSavePath = useRecoilValue(imageSavePathState)

  const savePreview = () => {
    if (removeBackgroundPreview) {
      const folderPath = path.join(imageSavePath, batchName, 'RemovedBackgrounds')
      if (!fs.existsSync(folderPath)) {
        fs.mkdirSync(folderPath)
      }
      const files = fs.readdirSync(folderPath)
      const fileNumber = files.filter((file) => file.startsWith(`${removeBackground}_`)).length + 1
      const fileName = `${removeBackground}_${fileNumber}.png`
      const filePath = path.join(folderPath, fileName)
      const data = removeBackgroundPreview.replace(/^data:image\/\w+;base64,/, '')
      const buffer = Buffer.from(data, 'base64')
      fs.writeFile(filePath, buffer, (err) => {
        if (err) {
          console.error(err)
        } else {
          toast({
            title: `Preview Saved to ${filePath}`,
            status: 'success',
            position: 'top',
            duration: 1500,
            isClosable: true,
          })
          console.log(`Preview saved as ${filePath}`)
        }
      })
    }
  }

  return (
    <Flex pt="5px" flexDirection="column" alignItems="center" justifyContent="center">
      <Box
        bg="#080B16"
        height="180px"
        width="180px"
        border="1px"
        borderStyle="ridge"
        rounded="md"
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          textAlign: 'center',
          borderColor: '#FFFFFF20',
        }}
      >
        <Image boxSize="178px" fit="contain" rounded="md" src={removeBackgroundPreview} />
      </Box>
      <Flex alignItems="center" justifyContent="center">
        <ButtonGroup pt="5px" isAttached variant="outline">
          <Tooltip label="Save Preview">
            <IconButton
              border="2px"
              icon={<FaSave />}
              onClick={savePreview}
              width="90px"
              aria-label="Save Preview"
            />
          </Tooltip>
          <Tooltip label="Clear Preview">
            <IconButton
              aria-label="Clear Preview"
              border="2px"
              icon={<FaTrashAlt />}
              width="90px"
              onClick={() => {
                setRemoveBackgroundPreview('')
              }}
            />
          </Tooltip>
        </ButtonGroup>
      </Flex>
    </Flex>
  )
}

export default RemoveBackgroundPreview

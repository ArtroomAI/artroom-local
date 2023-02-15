import React, { useCallback, useEffect, useRef, useState } from 'react'
import { useRecoilState } from 'recoil'
import * as atom from '../../atoms/atoms'
import {
  Box,
  Image as ChakraImage,
  IconButton,
  ButtonGroup
} from '@chakra-ui/react'
import { FiUpload } from 'react-icons/fi'
import { FaTrashAlt } from 'react-icons/fa'

const getImageDimensions = (base64: string) => {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () =>
      resolve({ width: img.naturalWidth, height: img.naturalHeight })
    img.onerror = reject
    img.src = base64
  })
}

const DragDropFile = () => {
  const [dragActive, setDragActive] = useState(false)
  const inputRef = useRef(null)
  const [imageSettings, setImageSettings] = useRecoilState(
    atom.imageSettingsState
  )
  const [initImagePath, setInitImagePath] = useRecoilState(
    atom.initImagePathState
  )
  const [aspectRatioSelection, setAspectRatioSelection] = useRecoilState(
    atom.aspectRatioSelectionState
  )

  useEffect(() => {
    if (initImagePath) {
      console.log(initImagePath)
      window.api.getImageFromPath(initImagePath).then(result => {
        getImageDimensions(result.b64).then(dimensions => {
          if (aspectRatioSelection === 'Init Image') {
            setImageSettings({
              ...imageSettings,
              init_image: result.b64,
              width: dimensions.width,
              height: dimensions.height
            })
          } else {
            setImageSettings({
              ...imageSettings,
              init_image: result.b64
            })
          }
        })
      })
    }
    // update only when ImagePath is changed - prevents changing settings infinitely
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initImagePath])

  // Handle drag events
  const handleDrag: React.DragEventHandler<HTMLElement> = function (e) {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  // Triggers when file is dropped
  const handleDrop: React.DragEventHandler<HTMLDivElement> = useCallback(e => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files)
    }
  }, [])

  const handleFile = useCallback((e: FileList) => {
    const file = e[0]
    if (
      file.type === 'image/jpeg' ||
      file.type === 'image/png' ||
      file.type === 'image/heic'
    ) {
      console.log(file.path)
      setInitImagePath(file.path)
    } else {
      console.log(
        'Invalid file type. Please select an image file (jpg, png or heic).'
      )
    }
  }, [])

  // Triggers when file is selected with click
  const handleChange: React.ChangeEventHandler<HTMLInputElement> = function (
    e
  ) {
    e.preventDefault()
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files)
    }
  }

  // Triggers the input when the button is clicked
  const onButtonClick = () => {
    inputRef.current.click()
  }

  return (
    <Box bg="#080B16" height="140px" width="140px">
      {imageSettings.init_image.length > 0 ? (
        <Box
          border="1px"
          borderStyle="ridge"
          height="140px"
          onClick={onButtonClick}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          rounded="md"
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            textAlign: 'center',
            borderColor: '#FFFFFF20'
          }}
          width="140px"
        >
          <ChakraImage
            boxSize="140px"
            fit="contain"
            rounded="md"
            src={imageSettings.init_image}
          />
        </Box>
      ) : (
        <Box
          border="1px"
          borderStyle="ridge"
          height="140px"
          onClick={onButtonClick}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          px={4}
          rounded="md"
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            textAlign: 'center',
            borderColor: '#FFFFFF20'
          }}
          width="140px"
        >
          Click or Drag Image Here
        </Box>
      )}

      <form
        id="form-file-upload"
        onDragEnter={handleDrag}
        onSubmit={e => e.preventDefault()}
      >
        <input
          accept="image/png, image/jpeg"
          id="input-file-upload"
          multiple={false}
          onChange={handleChange}
          ref={inputRef}
          type="file"
        />

        <label htmlFor="input-file-upload" id="label-file-upload">
          <ButtonGroup pt={2} isAttached variant="outline">
            <IconButton
              border="2px"
              icon={<FiUpload />}
              onClick={onButtonClick}
              width="100px"
              aria-label="upload"
            />
            <IconButton
              aria-label="Clear Init Image"
              border="2px"
              icon={<FaTrashAlt />}
              onClick={event => {
                setInitImagePath('')
                if (aspectRatioSelection === 'Init Image') {
                  setImageSettings({
                    ...imageSettings,
                    init_image: '',
                    aspect_ratio: 'None'
                  })
                  setAspectRatioSelection('None')
                } else {
                  setImageSettings({
                    ...imageSettings,
                    init_image: ''
                  })
                }
              }}
            />
          </ButtonGroup>
        </label>
      </form>
    </Box>
  )
}

export default DragDropFile

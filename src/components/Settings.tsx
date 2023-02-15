import React, { useState, useEffect, useContext, useCallback } from 'react'
import { useRecoilState } from 'recoil'
import * as atom from '../atoms/atoms'
import {
  Box,
  Button,
  Checkbox,
  Flex,
  FormControl,
  FormLabel,
  Input,
  VStack,
  HStack,
  Tooltip,
  Radio,
  RadioGroup,
  Stack,
  Select,
  NumberInput,
  NumberInputField,
  Spacer,
  useToast
} from '@chakra-ui/react'
import { FaQuestionCircle } from 'react-icons/fa'
import DebugInstallerModal from './Modals/DebugInstallerModal'
import { SocketContext, SocketOnEvents } from '../socket'

function Settings() {
  const toast = useToast({})
  const [imageSettings, setImageSettings] = useRecoilState(
    atom.imageSettingsState
  )
  const [long_save_path, setLongSavePath] = useRecoilState(
    atom.longSavePathState
  )
  const [highres_fix, setHighresFix] = useRecoilState(atom.highresFixState)

  const [debug_mode, setDebugMode] = useRecoilState(atom.debugMode)
  const [delay, setDelay] = useRecoilState(atom.delayState)
  const [downloadMessage, setDownloadMessage] = useState('')

  const [debug_mode_orig, setDebugModeOrig] = useState(true)

  const socket = useContext(SocketContext)

  useEffect(
    () => {
      window.api.getSettings().then(result => {
        const settings = JSON.parse(result)
        setImageSettings({
          ...imageSettings,
          speed: settings.speed,
          ckpt_dir: settings.ckpt_dir,
          save_grid: settings.save_grid,
          vae: settings.vae,
          image_save_path: settings.image_save_path
        })

        setLongSavePath(settings.long_save_path)
        setHighresFix(settings.highres_fix)
        setDelay(settings.delay)
        setDebugMode(settings.debug_mode)
        setDebugModeOrig(settings.debug_mode)
      })
    },
    // run only once
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  )

  const handleSaveSettingsAndRestart: SocketOnEvents['update_settings_with_restart'] =
    useCallback(
      data => {
        if (data.status === 'Success') {
          toast({
            title: 'Settings have been updated',
            status: 'success',
            position: 'top',
            duration: 1500,
            isClosable: false,
            containerStyle: {
              pointerEvents: 'none'
            }
          })
        } else {
          toast({
            title: 'Error during updating settings',
            description: data.status_message,
            status: 'error',
            position: 'top',
            duration: 1500,
            isClosable: false,
            containerStyle: {
              pointerEvents: 'none'
            }
          })
        }
        window.api.restartServer(debug_mode)
      },
      [debug_mode, toast]
    )

  const handleSaveSettings: SocketOnEvents['update_settings'] = useCallback(
    data => {
      if (data.status === 'Success') {
        toast({
          title: 'Settings have been updated',
          status: 'success',
          position: 'top',
          duration: 1500,
          isClosable: false,
          containerStyle: {
            pointerEvents: 'none'
          }
        })
      } else {
        toast({
          title: 'Error during updating settings',
          description: data.status_message,
          status: 'error',
          position: 'top',
          duration: 1500,
          isClosable: false,
          containerStyle: {
            pointerEvents: 'none'
          }
        })
      }
    },
    [toast]
  )

  const saveSettings = useCallback(() => {
    setDebugModeOrig(debug_mode)
    const output = {
      long_save_path,
      highres_fix,
      debug_mode,
      delay,
      speed: imageSettings.speed,
      image_save_path: imageSettings.image_save_path,
      save_grid: imageSettings.save_grid,
      vae: imageSettings.vae,
      ckpt_dir: imageSettings.ckpt_dir
    }
    socket.emit('update_settings', output)
  }, [
    debug_mode,
    delay,
    highres_fix,
    imageSettings.ckpt_dir,
    imageSettings.image_save_path,
    imageSettings.save_grid,
    imageSettings.speed,
    imageSettings.vae,
    long_save_path,
    socket
  ])

  const saveSettingsWithRestart = useCallback(() => {
    setDebugModeOrig(debug_mode)
    const output = {
      long_save_path,
      highres_fix,
      debug_mode,
      delay,
      speed: imageSettings.speed,
      image_save_path: imageSettings.image_save_path,
      save_grid: imageSettings.save_grid,
      vae: imageSettings.vae,
      ckpt_dir: imageSettings.ckpt_dir
    }
    socket.emit('update_settings_with_restart', output)
  }, [
    debug_mode,
    delay,
    highres_fix,
    imageSettings.ckpt_dir,
    imageSettings.image_save_path,
    imageSettings.save_grid,
    imageSettings.speed,
    imageSettings.vae,
    long_save_path,
    socket
  ])

  useEffect(() => {
    window.api.fixButtonProgress((_, str) => {
      setDownloadMessage(str)
      console.log(str)
    })
  }, [])

  // on socket message
  useEffect(() => {
    socket.on('update_settings', handleSaveSettings)
    socket.on('update_settings_with_restart', handleSaveSettingsAndRestart)

    return () => {
      socket.off('update_settings', handleSaveSettings)
      socket.off('update_settings_with_restart', handleSaveSettingsAndRestart)
    }
  }, [socket, handleSaveSettings, handleSaveSettingsAndRestart])

  const submitEvent = () => {
    console.log(debug_mode, debug_mode_orig)
    if (debug_mode === true && debug_mode_orig === false) {
      // Restart server and turn debug mode on
      toast({
        id: 'restart-server-debug-on',
        title: 'Resetting Artroom with Debug Mode on',
        status: 'info',
        position: 'top',
        duration: 7000,
        isClosable: true
      })
      saveSettingsWithRestart() // Restarts flask server first, then save settings
    } else if (debug_mode === false && debug_mode_orig === true) {
      // Restart server and turn debug mode off
      toast({
        id: 'restart-server-debug-off',
        title: 'Resetting Artroom with Debug Mode off',
        status: 'info',
        position: 'top',
        duration: 7000,
        isClosable: true
      })
      saveSettingsWithRestart() // Restarts flask server first, then save settings
    } else {
      // Just save settings as normal
      saveSettings()
    }
  }

  const chooseUploadPath = () => {
    window.api.chooseUploadPath().then(result => {
      setImageSettings({ ...imageSettings, image_save_path: result })
    })
  }

  const chooseCkptDir = () => {
    window.api.chooseUploadPath().then(result => {
      setImageSettings({ ...imageSettings, ckpt_dir: result })
    })
  }

  return (
    <Box height="90%" ml="30px" p={5} rounded="md" width="75%">
      <VStack align="flex-start" className="settings" spacing={5}>
        <FormControl className="image-save-path-input" width="full">
          <FormLabel htmlFor="image_save_path">Image Output Folder</FormLabel>

          <HStack>
            <Input
              id="image_save_path"
              name="image_save_path"
              onChange={event =>
                setImageSettings({
                  ...imageSettings,
                  image_save_path: event.target.value
                })
              }
              type="text"
              value={imageSettings.image_save_path}
              variant="outline"
            />

            <Button onClick={chooseUploadPath}>Choose</Button>
          </HStack>
        </FormControl>

        <FormControl className="model-ckpt-dir-input" width="full">
          <HStack>
            <Tooltip
              fontSize="md"
              label="When making folders, choose a non-root directory (so do E:/Models instead of E:/)"
              placement="top"
              shouldWrapChildren
            >
              <FaQuestionCircle color="#777" />
            </Tooltip>

            <FormLabel htmlFor="ckpt_dir">Model Weights Folder</FormLabel>
          </HStack>

          <HStack>
            <Input
              id="ckpt_dir"
              name="ckpt_dir"
              onChange={event =>
                setImageSettings({
                  ...imageSettings,
                  ckpt_dir: event.target.value
                })
              }
              type="text"
              value={imageSettings.ckpt_dir}
              variant="outline"
            />

            <Button onClick={chooseCkptDir}>Choose</Button>
          </HStack>
        </FormControl>

        <FormControl className="speed-input">
          <HStack>
            <Tooltip
              fontSize="md"
              label="Generate faster but use more GPU memory. Be careful of OOM (Out of Memory) error"
              mt="3"
              placement="right"
              shouldWrapChildren
            >
              <FaQuestionCircle color="#777" />
            </Tooltip>

            <FormLabel htmlFor="Speed">Choose Generation Speed</FormLabel>
          </HStack>

          <RadioGroup
            id="speed"
            name="speed"
            onChange={value =>
              setImageSettings({ ...imageSettings, speed: value })
            }
            value={imageSettings.speed}
          >
            <Stack direction="row" spacing="20">
              ``
              <Radio value="Low">Low</Radio>
              <Radio value="Medium">Medium</Radio>
              <Radio value="High">High</Radio>
            </Stack>
          </RadioGroup>
        </FormControl>

        <FormControl className="queue-delay-input">
          <HStack>
            <Tooltip
              fontSize="md"
              label="Add a delay between runs. Mostly to prevent GPU from getting too hot if it's nonstop. Runs on next Queue start (need to Stop and Start queue again)."
              placement="top"
              shouldWrapChildren
            >
              <FaQuestionCircle color="#777" />
            </Tooltip>

            <FormLabel htmlFor="delay">Queue Delay</FormLabel>
          </HStack>

          <NumberInput
            id="delay"
            min={1}
            name="delay"
            onChange={(v, n) => {
              setDelay(n)
            }}
            step={1}
            value={delay}
            variant="outline"
            w="150px"
          >
            <NumberInputField id="delay" />
          </NumberInput>
        </FormControl>

        <HStack className="highres-fix-input">
          <Checkbox
            id="highres_fix"
            isChecked={highres_fix}
            name="highres_fix"
            onChange={() => {
              setHighresFix(!highres_fix)
            }}
          >
            Use Highres Fix
          </Checkbox>

          <Tooltip
            fontSize="md"
            label="Once you get past 1024x1024, will generate a smaller image, upscale, and then img2img to improve quality"
            mt="3"
            placement="right"
            shouldWrapChildren
          >
            <FaQuestionCircle color="#777" />
          </Tooltip>
        </HStack>

        <HStack className="long-save-path-input">
          <Checkbox
            id="long_save_path"
            isChecked={long_save_path}
            name="long_save_path"
            onChange={() => {
              setLongSavePath(!long_save_path)
            }}
          >
            Use Long Save Path
          </Checkbox>

          <Tooltip
            fontSize="md"
            label="Put the images in a folder named after the prompt (matches legacy Artroom naming conventions)"
            mt="3"
            placement="right"
            shouldWrapChildren
          >
            <FaQuestionCircle color="#777" />
          </Tooltip>
        </HStack>

        <HStack className="save-grid-input">
          <Checkbox
            id="save_grid"
            isChecked={imageSettings.save_grid}
            name="save_grid"
            onChange={() => {
              setImageSettings({
                ...imageSettings,
                save_grid: !imageSettings.save_grid
              })
            }}
          >
            Save Grid
          </Checkbox>

          <Tooltip
            fontSize="md"
            label="Save batch in one big grid image"
            mt="3"
            placement="right"
            shouldWrapChildren
          >
            <FaQuestionCircle color="#777" />
          </Tooltip>
        </HStack>

        <HStack className="debug-mode-input">
          <Checkbox
            id="debug_mode"
            isChecked={debug_mode}
            name="debug_mode"
            onChange={() => {
              setDebugMode(!debug_mode)
            }}
          >
            Debug Mode
          </Checkbox>

          <Tooltip
            fontSize="md"
            label="Opens cmd console of detailed outputs during image generation"
            mt="3"
            placement="right"
            shouldWrapChildren
          >
            <FaQuestionCircle color="#777" />
          </Tooltip>
        </HStack>

        <Flex width="100%">
          <Button
            alignContent="center"
            className="save-settings-button"
            onClick={submitEvent}
          >
            Save Settings
          </Button>
          <Spacer />
          <DebugInstallerModal />
          <Button
            marginLeft={1}
            backgroundColor="red.600"
            colorScheme="red"
            alignContent="center"
            className="reinstall-python-dependencies"
            onClick={window.api.pythonInstallDependencies}
          >
            Update Packages
          </Button>
        </Flex>

        {downloadMessage && (
          <Flex width="100%">
            <Flex width="100%">Installation progress</Flex>
            <Spacer />
            <Flex width="100%">{downloadMessage}</Flex>
          </Flex>
        )}
      </VStack>
    </Box>
  )
}

export default Settings

import React, { useEffect, useState } from 'react'
import { useRecoilState, useRecoilValue } from 'recoil'
import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  Flex,
  Button,
  Input,
  Text,
  useToast,
  Checkbox,
  Link,
  VStack,
  RadioGroup,
  Radio,
  HStack,
  Box,
} from '@chakra-ui/react'
import {
  artroomPathState,
  debugModeState,
  modelsDirState,
  cloudOnlyState,
  gpuTypeState,
} from './SettingsManager'
import path from 'path'
import gpuInfo from 'gpu-info'

export const InstallerManager = () => {
  const toast = useToast({})
  const [gpuSelection, setGpuSelection] = useState<string>('NVIDIA')
  const [gpuType, setGpuType] = useRecoilState(gpuTypeState)
  const debugMode = useRecoilValue(debugModeState)

  const [cloudOnly, setCloudOnly] = useRecoilState(cloudOnlyState)

  const [showArtroomInstaller, setShowArtroomInstaller] = useState(false)
  const [artroomPath, setArtroomPath] = useRecoilState(artroomPathState)
  const [modelsDir, setModelsDir] = useRecoilState(
    modelsDirState || path.join(artroomPath, 'artroom', 'model_weights')
  )
  const [customPath, setCustomPath] = useState(false)
  const [downloadMessage, setDownloadMessage] = useState('')
  const [downloading, setDownloading] = useState(false)

  const [starterModels, setStarterModels] = useState({
    SDXL: true,
    ChilloutMix: false,
    Counterfeit: false,
    DreamShaper: false,
  })

  const detectGPU = async () => {
    try {
      const info = await gpuInfo()
      const gpuVendor = info[0].VideoProcessor.toLowerCase()
      if (gpuVendor.includes('nvidia')) {
        console.log('DETECTED NVIDIA')
        setGpuSelection('NVIDIA')
      } else if (gpuVendor.includes('amd')) {
        console.log('DETECTED AMD')
        setGpuSelection('AMD')
      }
    } catch (err) {
      console.error('Failed to detect GPU:', err)
    }
  }

  useEffect(() => {
    detectGPU()
  }, [])

  useEffect(() => {
    if (cloudOnly) return
    window.api.runPyTests(artroomPath).then((result) => {
      if (result === 'success\r\n' || process.platform == 'linux') {
        console.log(result)
        window.api.pythonInstallDependencies(artroomPath, gpuSelection).then(() => {
          setGpuType(gpuSelection)
          window.api.startArtroom(artroomPath, debugMode)
          setShowArtroomInstaller(false)
          toast({
            title: 'All Artroom paths & dependencies successfully found!',
            status: 'success',
            position: 'top',
            duration: 2000,
            isClosable: true,
          })
        })
      } else if (result.length > 0) {
        setShowArtroomInstaller(true)
      }
    })
  }, [artroomPath])

  useEffect(() => {
    const handlerDiscard = window.api.fixButtonProgress((_: any, str: string) => {
      setDownloadMessage(str)
    })

    return () => {
      handlerDiscard()
    }
  }, [])

  const handleRunClick = async () => {
    toast({
      title: `Installing Artroom Backend for ${gpuSelection} `,
      status: 'success',
      position: 'top',
      duration: 5000,
      isClosable: true,
      containerStyle: {
        pointerEvents: 'none',
      },
    })
    setDownloading(true)
    try {
      let dir = modelsDir
      if (!customPath) {
        dir = path.join(artroomPath, 'artroom', 'model_weights')
        setModelsDir(dir)
      }
      await window.api.backupDownload(artroomPath, gpuSelection)
      await window.api.downloadStarterModels(dir, starterModels)
      setDownloading(false)
      setShowArtroomInstaller(false)
      window.api.startArtroom(artroomPath, debugMode)
    } catch (error) {
      console.error(error)
      setDownloading(false)
    }
  }

  function handleSelectArtroomClick() {
    window.api.chooseUploadPath().then(setArtroomPath)
  }
  function handleSelectModelClick() {
    window.api.chooseUploadPath().then(setModelsDir)
  }

  return (
    <Modal
      size="4xl"
      isOpen={showArtroomInstaller}
      onClose={() => {}} // Disable closing the modal
      scrollBehavior="outside"
      isCentered
    >
      <ModalOverlay bg="blackAlpha.900" />
      <ModalContent>
        <ModalHeader>{`Artroom Installer (~6GB)`}</ModalHeader>
        <ModalBody>
          <Text>Artroom Engine Backend Install Location</Text>
          {customPath && (
            <Text mb="4">{`(Note: Do NOT select the Artroom folder that was installed on startup)`}</Text>
          )}
          <Flex flexDirection="row" justifyItems="center" alignItems="center" mb="4">
            <Input
              width="80%"
              placeholder="Artroom will be saved in YourPath/artroom"
              value={artroomPath}
              onChange={(event) => {
                setArtroomPath(event.target.value)
              }}
              isDisabled={!customPath}
              mr="4"
            />
            {customPath && <Button onClick={handleSelectArtroomClick}>Select</Button>}
          </Flex>
          <Text mb="1">{`Model Path (This can be changed later in Settings)`}</Text>
          <Flex flexDirection="row" alignItems="center" mb="4">
            <Input
              width="80%"
              placeholder="Model will be saved in YourPath/artroom/model_weights"
              value={modelsDir}
              onChange={(event) => {
                setModelsDir(event.target.value)
              }}
              isDisabled={!customPath}
              mr="4"
            />
            {customPath && <Button onClick={handleSelectModelClick}>Select</Button>}
          </Flex>
          <Checkbox
            mb="4"
            isChecked={customPath}
            onChange={() => {
              setCustomPath(!customPath)
            }}
          >
            Use Custom Path
          </Checkbox>
          <VStack alignItems="start" mb="4">
            <Text>{`Select GPU (If unsure, use preselected):`}</Text>
            <RadioGroup value={gpuSelection} onChange={setGpuSelection}>
              <HStack spacing="24px">
                <Radio value="NVIDIA">NVIDIA</Radio>
                <Radio value="AMD">AMD</Radio>
              </HStack>
            </RadioGroup>
          </VStack>
          <Text>{`Do you want a starter model (optional)?`}</Text>
          <Text mb="4">
            {`Want more? Download them to your Artroom folder from `}
            <Link color="blue.500" href="https://civitai.com/" isExternal>
              civitai.com
            </Link>
          </Text>
          <VStack alignItems="flex-start">
            <Checkbox
              isChecked={starterModels.SDXL}
              onChange={() => {
                setStarterModels((prevState) => ({
                  ...prevState,
                  SDXL: !prevState.SDXL,
                }))
              }}
            >
              {`SDXL`}
            </Checkbox>
            ;
            <Checkbox
              isChecked={starterModels.DreamShaper}
              onChange={() => {
                setStarterModels((prevState) => ({
                  ...prevState,
                  DreamShaper: !prevState.DreamShaper,
                }))
              }}
            >
              {`(General) DreamShaper SDXL`}
            </Checkbox>
            <Checkbox
              isChecked={starterModels.ChilloutMix}
              onChange={() => {
                setStarterModels((prevState) => ({
                  ...prevState,
                  ChilloutMix: !prevState.ChilloutMix,
                }))
              }}
            >
              {`(Realistic) ChilloutMix`}
            </Checkbox>
            <Checkbox
              isChecked={starterModels.Counterfeit}
              onChange={() => {
                setStarterModels((prevState) => ({
                  ...prevState,
                  Counterfeit: !prevState.Counterfeit,
                }))
              }}
            >
              {`(Anime) Counterfeit`}
            </Checkbox>
            <Text
              _hover={{
                cursor: 'pointer',
                textDecoration: 'underline',
              }}
              onClick={() => {
                window.api.openInstallTutorial()
              }}
            >
              {`Having Trouble Downloading? Click Here to get manual install instructions (Don't worry, it's easy)`}
            </Text>
          </VStack>
          {downloadMessage && (
            <Flex width="100%" justifyContent="space-between">
              <Text>Installation progress</Text>
              <Text pr="80px">{downloadMessage}</Text>
            </Flex>
          )}
        </ModalBody>
        <ModalFooter justifyContent="center">
          <Button
            isLoading={downloading}
            isDisabled={downloading}
            onClick={() => {
              handleRunClick()
            }}
          >
            Install Artroom
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  )
}

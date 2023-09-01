import React, { useState, useCallback, useContext } from 'react'
import {
  Box,
  Button,
  ButtonGroup,
  IconButton,
  FormControl,
  FormLabel,
  Input,
  VStack,
  HStack,
  Tooltip,
  NumberInputField,
  NumberInput,
  Text,
  Stack,
  Select,
  useToast,
  Flex,
  Icon,
  Accordion,
  AccordionButton,
  AccordionItem,
  AccordionIcon,
  AccordionPanel,
  GridItem,
  Grid,
  Divider,
  RadioGroup,
  Radio,
} from '@chakra-ui/react'
import { FaFolder, FaQuestionCircle, FaTrashAlt } from 'react-icons/fa'
import { SocketContext } from '../socket'
import { useDropzone } from 'react-dropzone'
import { useRecoilValue } from 'recoil'
import { imageSavePathState, modelsDirState, modelsState } from '../SettingsManager'
import path from 'path'
import Authentication from './Authentication/Authentication'

function Trainer() {
  const toast = useToast({})
  const [images, setImages] = useState<string[]>([])
  const [name, setName] = useState('')
  const [trigger_word, setTriggerWord] = useState('')
  const [model, setModel] = useState('')

  const imageSavePath = useRecoilValue(imageSavePathState)
  const modelsDir = useRecoilValue(modelsDirState)
  const models = useRecoilValue(modelsState)
  const socket = useContext(SocketContext)

  const [resolutionWidth, setResolutionWidth] = useState(512)
  const [resolutionHeight, setResolutionHeight] = useState(512)
  const [numRepeats, setNumRepeats] = useState(10)
  const [maxTrainSteps, setMaxTrainSteps] = useState(1000)
  const [saveEveryNEpochs, setSaveEveryNEpochs] = useState(1)
  const [trainBatchSize, setTrainBatchSize] = useState(1)
  const [networkAlpha, setNetworkAlpha] = useState(128)
  const [textEncoderLr, setTextEncoderLr] = useState(0.00005)
  const [unetLr, setUnetLr] = useState(0.0001)
  const [networkDim, setNetworkDim] = useState(64)
  const [lrSchedulerNumCycles, setLrSchedulerNumCycles] = useState(1)
  const [learningRate, setLearningRate] = useState(0.0001)
  const [lrScheduler, setLrScheduler] = useState('constant')
  const [optimizerType, setOptimizerType] = useState('AdamW')
  const [bucketResoSteps, setBucketResoSteps] = useState(64)
  const [minBucketReso, setMinBucketReso] = useState(384)
  const [maxBucketReso, setMaxBucketReso] = useState(1024)
  const [activePreset, setActivePreset] = useState('None')

  const handleRadioChange = (preset: string) => {
    setActivePreset(preset)
    if (preset == 'SDXL') {
      handleSDXLPresets()
    }
    if (preset == 'SDv1.5') {
      handleSDv1Presets()
    }
  }

  const handleSDv1Presets = () => {
    setResolutionWidth(512)
    setResolutionHeight(512)
    setNumRepeats(10)
    setMaxTrainSteps(1000)
    setSaveEveryNEpochs(2)
    setTrainBatchSize(1)
    setNetworkAlpha(1)
    setTextEncoderLr(0.0004)
    setUnetLr(0.0004)
    setNetworkDim(32)
    setLrSchedulerNumCycles(1)
    setLearningRate(0.0004)
    setLrScheduler('constant')
    setOptimizerType('Adafactor')
    setBucketResoSteps(64)
    setMinBucketReso(256)
    setMaxBucketReso(1024)
  }
  0.0004
  const handleSDXLPresets = () => {
    setResolutionWidth(1024)
    setResolutionHeight(1024)
    setNumRepeats(10)
    setMaxTrainSteps(500)
    setSaveEveryNEpochs(1)
    setTrainBatchSize(1)
    setNetworkAlpha(1)
    setNetworkDim(32)
    setTextEncoderLr(0.0004)
    setUnetLr(0.0004)
    setLrSchedulerNumCycles(1)
    setLearningRate(0.0004)
    setLrScheduler('constant')
    setOptimizerType('Adafactor')
    setBucketResoSteps(64)
    setMinBucketReso(512)
    setMaxBucketReso(2048)
  }

  const chooseUploadPath = () => {
    window.api.chooseImages().then(setImages)
  }

  const goToModelFolder = useCallback(() => {
    if (modelsDir !== '') {
      window.api.showInExplorer(modelsDir)
    }
  }, [modelsDir])

  const Dropzone = () => {
    const toast = useToast()
    const onDrop = useCallback(
      (acceptedFiles: File[]) => {
        setImages(acceptedFiles.map((file) => file.path))

        toast({
          title: 'Files added!',
          status: 'success',
          duration: 5000,
          isClosable: true,
        })
      },
      [toast]
    )

    const { getRootProps, getInputProps } = useDropzone({
      onDrop,
      useFsAccessApi: false,
    })

    return (
      <Box
        borderWidth="1px"
        borderRadius="md"
        p="5"
        my="5"
        backgroundColor="#080B16"
        width="100%"
        {...getRootProps()}
      >
        <input multiple {...getInputProps()} />
        <Flex align="center" justify="center" height="150px">
          <Icon name="cloud-upload" />
          <Text ml="2">Drop your files here or click to browse</Text>
        </Flex>
      </Box>
    )
  }

  const startTraining = useCallback(() => {
    const sendInvalidToast = (text: string) => {
      toast({
        title: 'Invalid Input',
        description: `${text}`,
        status: 'error',
        position: 'top',
        duration: 2000,
        isClosable: false,
        containerStyle: {
          pointerEvents: 'none',
        },
      })
    }
    const checkValidity = (output: any) => {
      if (!output['images'].length) {
        sendInvalidToast('Please select at least 1 image')
        return false
      }
      if (!model.length) {
        sendInvalidToast('Please select a model')
        return false
      }
      if (!output['name'].length) {
        sendInvalidToast('Please select a name')
        return false
      }
      if (parseInt(output['minBucketReso']) > parseInt(output['maxBucketReso'])) {
        sendInvalidToast('Min Bucket Resolution must be at less than Max Bucket Resolution')
        return false
      }
      return true
    }
    const output = {
      images,
      name,
      trigger_word,
      image_save_path: imageSavePath,
      model: path.join(modelsDir, model),
      modelsDir,
      resolution: `${resolutionWidth},${resolutionHeight}`,
      numRepeats: `${numRepeats}`,
      networkAlpha: `${networkAlpha}`,
      maxTrainSteps: `${maxTrainSteps}`,
      textEncoderLr: `${textEncoderLr}`,
      unetLr: `${unetLr}`,
      networkDim: `${networkDim}`,
      lrSchedulerNumCycles: `${lrSchedulerNumCycles}`,
      learningRate: `${learningRate}`,
      lrScheduler: `${lrScheduler}`,
      trainBatchSize: `${trainBatchSize}`,
      saveEveryNEpochs: `${saveEveryNEpochs}`,
      optimizerType: `${optimizerType}`,
      bucketResoSteps: `${bucketResoSteps}`,
      minBucketReso: `${minBucketReso}`,
      maxBucketReso: `${maxBucketReso}`,
    }
    const valid = checkValidity(output)
    if (valid) {
      toast({
        title: 'Received!',
        description: `You'll get a notification 🔔 when your lora is ready! (First time training make take longer while it installs the necessary packages)`,
        status: 'success',
        position: 'top',
        duration: 3000,
        isClosable: false,
        containerStyle: {
          pointerEvents: 'none',
        },
      })
      socket.emit('train', output)
    }
  }, [socket, toast, trigger_word, images, imageSavePath, modelsDir])

  return (
    <Box height="90%" ml="30px" p={4} rounded="md" width="90%">
      <form>
        <VStack align="flex-start" className="upscale" spacing={5}>
          <Dropzone />
          <FormControl className="upscale-images-input" width="full">
            <HStack>
              <Tooltip
                fontSize="md"
                label="You can select all images in folder with Ctrl+A, it will ignore folders in upscaling"
                mt="3"
                placement="right"
                shouldWrapChildren
              >
                <FaQuestionCircle color="#777" />
              </Tooltip>

              <FormLabel htmlFor="images">Choose images to train</FormLabel>
            </HStack>

            <HStack>
              <Input
                id="images"
                name="images"
                onChange={(event) =>
                  setImages(event.target.value.split(',')?.filter((e) => e !== ''))
                }
                type="text"
                value={images}
                variant="outline"
              />

              <ButtonGroup isAttached pl="10px" variant="outline">
                <Button onClick={chooseUploadPath}>Choose</Button>

                <IconButton
                  aria-label="Clear Upscale Images"
                  icon={<FaTrashAlt />}
                  onClick={() => setImages([])}
                />
              </ButtonGroup>
            </HStack>
          </FormControl>

          <FormControl className="name" width="full">
            <FormLabel htmlFor="name">{`File Name`}</FormLabel>
            <HStack>
              <Input
                id="name"
                name="name"
                onChange={(event) => setName(event.target.value)}
                type="text"
                value={name}
                variant="outline"
              />
            </HStack>
          </FormControl>

          <FormControl className="trigger_word" width="full">
            <FormLabel htmlFor="trigger_word">{`Trigger Word (Optional)`}</FormLabel>
            <HStack>
              <Input
                id="trigger_word"
                name="trigger_word"
                onChange={(event) => setTriggerWord(event.target.value)}
                type="text"
                value={trigger_word}
                variant="outline"
              />
            </HStack>
          </FormControl>

          <FormControl className="model-ckpt-input">
            <FormLabel htmlFor="Ckpt">
              <HStack>
                <Text>Model</Text>
                <Button
                  fontSize="xs"
                  leftIcon={<FaFolder />}
                  _hover={{ cursor: 'pointer' }}
                  onClick={goToModelFolder}
                  variant="outline"
                  size="xs"
                >
                  View
                </Button>
              </HStack>
            </FormLabel>

            <Select
              id="ckpt"
              name="ckpt"
              onChange={(event) => setModel(event.target.value)}
              value={model}
              variant="outline"
            >
              <option value="">Choose Your Base Model</option>

              {models.ckpts.map((ckpt_option, i) => (
                <option key={i} value={ckpt_option}>
                  {ckpt_option}
                </option>
              ))}
            </Select>
          </FormControl>
          <RadioGroup onChange={handleRadioChange} value={activePreset}>
            <Stack direction="row" spacing={4}>
              <Radio value={'None'} colorScheme={activePreset === 'None' ? 'blue' : 'gray'}>
                Custom
              </Radio>
              <Radio value={'SDv1.5'} colorScheme={activePreset === 'SDv1.5' ? 'blue' : 'gray'}>
                SDv1.5 Preset
              </Radio>
              <Radio value={'SDXL'} colorScheme={activePreset === 'SDXL' ? 'blue' : 'gray'}>
                SDXL Preset
              </Radio>
            </Stack>
          </RadioGroup>
          <Accordion width="100%" allowToggle borderWidth={0} borderRadius={0}>
            <AccordionItem borderWidth={0} borderRadius={0}>
              <AccordionButton borderWidth={0} borderRadius={0}>
                <Box flex="1" textAlign="left">
                  Advanced Settings
                </Box>
                <AccordionIcon />
              </AccordionButton>
              <AccordionPanel pb={4}>
                <Grid templateColumns="repeat(2, 1fr)" gap={6}>
                  <GridItem>
                    <FormControl>
                      <FormLabel>Resolution Width</FormLabel>
                      <NumberInput
                        min={512}
                        max={1024}
                        step={64}
                        value={resolutionWidth}
                        onChange={(value) => setResolutionWidth(parseInt(value))}
                      >
                        <NumberInputField />
                      </NumberInput>
                    </FormControl>
                  </GridItem>
                  <GridItem>
                    <FormControl>
                      <FormLabel>Resolution Height</FormLabel>
                      <NumberInput
                        min={512}
                        max={1024}
                        step={64}
                        value={resolutionHeight}
                        onChange={(value) => setResolutionHeight(parseInt(value))}
                      >
                        <NumberInputField />
                      </NumberInput>
                    </FormControl>
                  </GridItem>
                  <GridItem>
                    <FormControl>
                      <FormLabel>Max Training Steps</FormLabel>
                      <NumberInput
                        min={1}
                        value={maxTrainSteps}
                        onChange={(value) => setMaxTrainSteps(parseInt(value))}
                      >
                        <NumberInputField />
                      </NumberInput>
                    </FormControl>
                  </GridItem>
                  <GridItem>
                    <FormControl>
                      <FormLabel>{`Save Every N Iterations (Epochs)`}</FormLabel>
                      <NumberInput
                        min={1}
                        value={saveEveryNEpochs}
                        onChange={(value) => setSaveEveryNEpochs(parseInt(value))}
                      >
                        <NumberInputField />
                      </NumberInput>
                    </FormControl>
                  </GridItem>
                  <GridItem>
                    <FormControl>
                      <FormLabel>Num Repeats of Each Image per Batch</FormLabel>
                      <NumberInput
                        min={1}
                        value={numRepeats}
                        onChange={(value) => setNumRepeats(parseInt(value))}
                      >
                        <NumberInputField />
                      </NumberInput>
                    </FormControl>
                  </GridItem>
                  <GridItem>
                    <FormControl>
                      <FormLabel>Training Batch Size</FormLabel>
                      <NumberInput
                        min={1}
                        value={trainBatchSize}
                        onChange={(value) => setTrainBatchSize(parseInt(value))}
                      >
                        <NumberInputField />
                      </NumberInput>
                    </FormControl>
                  </GridItem>
                  <GridItem colSpan={2}>
                    <Divider />
                  </GridItem>
                  <GridItem>
                    <FormControl>
                      <FormLabel>Network Alpha</FormLabel>
                      <NumberInput
                        min={0}
                        max={1024}
                        value={networkAlpha}
                        onChange={(value) => setNetworkAlpha(parseInt(value))}
                      >
                        <NumberInputField />
                      </NumberInput>
                    </FormControl>
                  </GridItem>
                  <GridItem>
                    <FormControl>
                      <FormLabel>Text Encoder Learning Rate</FormLabel>
                      <NumberInput
                        min={0}
                        max={1}
                        value={textEncoderLr}
                        onChange={(value) => setTextEncoderLr(parseFloat(value))}
                      >
                        <NumberInputField />
                      </NumberInput>
                    </FormControl>
                  </GridItem>
                  <GridItem>
                    <FormControl>
                      <FormLabel>UNet Learning Rate</FormLabel>
                      <NumberInput
                        min={0}
                        max={1}
                        value={unetLr}
                        onChange={(value) => setUnetLr(parseFloat(value))}
                      >
                        <NumberInputField />
                      </NumberInput>
                    </FormControl>
                  </GridItem>
                  <GridItem>
                    <FormControl>
                      <FormLabel>Network Dimension</FormLabel>
                      <NumberInput
                        min={0}
                        max={1024}
                        step={4}
                        value={networkDim}
                        onChange={(value) => setNetworkDim(parseInt(value))}
                      >
                        <NumberInputField />
                      </NumberInput>
                    </FormControl>
                  </GridItem>
                  <GridItem>
                    <FormControl>
                      <FormLabel>LR Scheduler Number of Cycles</FormLabel>
                      <NumberInput
                        min={0}
                        step={1}
                        value={lrSchedulerNumCycles}
                        onChange={(value) => setLrSchedulerNumCycles(parseInt(value))}
                      >
                        <NumberInputField />
                      </NumberInput>
                    </FormControl>
                  </GridItem>

                  <GridItem>
                    <FormControl>
                      <FormLabel>LR Scheduler</FormLabel>
                      <Select value={lrScheduler} onChange={(e) => setLrScheduler(e.target.value)}>
                        <option value="constant">Constant</option>
                        <option value="linear">Linear</option>
                        <option value="cosine">Cosine</option>
                        <option value="cosine_with_restarts">Cosine with restarts</option>
                        <option value="polynomial">Polynomial</option>
                        <option value="constant_with_warmup">Constant with warmup</option>
                        <option value="adafactor">Adafactor</option>
                      </Select>
                    </FormControl>
                  </GridItem>
                  <GridItem>
                    <FormControl>
                      <FormLabel>Optimizer Type</FormLabel>
                      <Select
                        value={optimizerType}
                        onChange={(e) => setOptimizerType(e.target.value)}
                      >
                        <option value="Prodigy">Prodigy</option>
                        <option value="AdamW">AdamW</option>
                        <option value="AdamW8bit">AdamW8bit</option>
                        <option value="PagedAdamW8bit">PagedAdamW8bit</option>
                        <option value="Lion">Lion</option>
                        <option value="SGDNesterov">SGDNesterov</option>
                        <option value="SGDNesterov8bit">SGDNesterov8bit</option>
                        <option value="Lion8bit">Lion8bit</option>
                        <option value="PagedLion8bit">PagedLion8bit</option>
                        <option value="DAdaptation(DAdaptAdamPreprint)">
                          DAdaptation(DAdaptAdamPreprint)
                        </option>
                        <option value="DAdaptAdaGrad">DAdaptAdaGrad</option>
                        <option value="DAdaptAdam">DAdaptAdam</option>
                        <option value="DAdaptAdan">DAdaptAdan</option>
                        <option value="DAdaptAdanIP">DAdaptAdanIP</option>
                        <option value="DAdaptLion">DAdaptLion</option>
                        <option value="DAdaptSGD">DAdaptSGD</option>
                        <option value="Adafactor">Adafactor</option>
                      </Select>
                    </FormControl>
                  </GridItem>
                  <GridItem>
                    <FormControl>
                      <FormLabel>Learning Rate</FormLabel>
                      <NumberInput
                        min={0}
                        max={1}
                        value={learningRate}
                        onChange={(value) => setLearningRate(parseFloat(value))}
                      >
                        <NumberInputField />
                      </NumberInput>
                    </FormControl>
                  </GridItem>
                  <GridItem>
                    <FormControl>
                      <FormLabel>Bucket Reso Steps</FormLabel>
                      <NumberInput
                        min={64}
                        step={64}
                        value={bucketResoSteps}
                        onChange={(value) => setBucketResoSteps(parseInt(value))}
                      >
                        <NumberInputField />
                      </NumberInput>
                    </FormControl>
                  </GridItem>
                  <GridItem>
                    <FormControl>
                      <FormLabel>Min Bucket Resolution</FormLabel>
                      <NumberInput
                        min={64}
                        step={64}
                        value={minBucketReso}
                        onChange={(value) => setMinBucketReso(parseInt(value))}
                      >
                        <NumberInputField />
                      </NumberInput>
                    </FormControl>
                  </GridItem>
                  <GridItem>
                    <FormControl>
                      <FormLabel>Max Bucket Resolution</FormLabel>
                      <NumberInput
                        min={64}
                        step={64}
                        value={maxBucketReso}
                        onChange={(value) => setMaxBucketReso(parseInt(value))}
                      >
                        <NumberInputField />
                      </NumberInput>
                    </FormControl>
                  </GridItem>
                </Grid>
              </AccordionPanel>
            </AccordionItem>
          </Accordion>

          <Button alignSelf="center" className="train-button" onClick={startTraining}>
            Start Training
          </Button>
        </VStack>
      </form>
    </Box>
  )
}

const AuthenticatedTrainer = () => {
  return (
    <Authentication Component={Trainer} correctPassword="TEST" authenticatedKey={'LoraTraining'} />
  )
}

export default AuthenticatedTrainer

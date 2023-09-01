import React from 'react'
import { useRecoilState, useRecoilValue, useSetRecoilState } from 'recoil'
import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalBody,
  ModalFooter,
  Flex,
  Box,
  Text,
  Button,
  Icon,
  Divider,
  useToast,
} from '@chakra-ui/react'
import { BiImages, BiCopy } from 'react-icons/bi'
import { GoSettings } from 'react-icons/go'
import { AiFillFolderOpen } from 'react-icons/ai'
import * as atom from '../../atoms/atoms'
import { initImageState } from '../../SettingsManager'
import ImageObj from '../Reusable/ImageObj'
import { initImagePathState } from '../../atoms/atoms'

const ImageModalField = ({ data, header }: { data: string | number | Lora[]; header: string }) => {
  const toast = useToast({})
  const copyToClipboard = () => {
    window.api.copyToClipboard(`${data}`, 'text').then(() => {
      toast({
        title: 'Copied to clipboard',
        status: 'success',
        position: 'top',
        duration: 500,
      })
    })
  }

  const parseArray = (arr: Lora[]) => {
    return arr.map((el) => `${el.name} : ${el.weight}`).join('\n')
  }

  const toDisplay = Array.isArray(data) ? parseArray(data) : data

  return (
    <>
      <Box display="flex" alignItems="center" mt={2}>
        <Button borderRadius="10" variant="ghost" p={0} m={0} size="sm" onClick={copyToClipboard}>
          <Icon as={BiCopy} />
        </Button>
        <Text fontWeight="bold" color="white.800">
          {header}
        </Text>
      </Box>
      <Text pl="8">{toDisplay}</Text>
    </>
  )
}

function ImageModal({ imagePath }: { imagePath: string }) {
  const toast = useToast({
    status: 'success',
    position: 'top',
    duration: 500,
    isClosable: false,
    containerStyle: {
      pointerEvents: 'none',
    },
  })
  const imageModalB64 = useRecoilValue(atom.imageModalB64State)
  const imageModalMetadata = useRecoilValue(atom.imageModalMetadataState)
  const [showImageModal, setShowImageModal] = useRecoilState(atom.showImageModalState)
  const setInitImagePath = useSetRecoilState(initImagePathState)
  const setInitImage = useSetRecoilState(initImageState)
  console.log(imageModalMetadata)
  function handleClose() {
    setShowImageModal(false)
  }

  return (
    <Modal
      size="6xl"
      isOpen={showImageModal}
      onClose={handleClose}
      scrollBehavior="outside"
      blockScrollOnMount={false}
    >
      <ModalOverlay bg="blackAlpha.900" />
      <ModalContent>
        <ModalBody>
          <Flex>
            <Box w="100%">
              <ImageObj b64={imageModalB64} path={imagePath} active />
              <Button
                borderRadius="10"
                variant="ghost"
                colorScheme="blue"
                mr={2}
                fontSize="14"
                fontWeight="normal"
                onClick={() => {
                  window.api.copyToClipboard(imageModalB64)
                  toast({ title: 'copied to clipboard' })
                }}
              >
                <Icon mr={2} as={BiImages} />
                Copy Image
              </Button>
              <Button
                borderRadius="10"
                variant="ghost"
                colorScheme="blue"
                mr={2}
                fontSize="14"
                fontWeight="normal"
                onClick={() => {
                  setInitImagePath(imagePath)
                  setInitImage(imageModalB64)
                  toast({ title: 'init image set' })
                }}
              >
                <Icon mr={2} as={BiImages} />
                Set Starting Image
              </Button>
              <Button
                borderRadius="10"
                variant="ghost"
                colorScheme="blue"
                mr={2}
                fontSize="14"
                fontWeight="normal"
                onClick={() => window.api.showItemInFolder(imagePath)}
              >
                <Icon mr={2} as={AiFillFolderOpen} />
                View in Files
              </Button>
            </Box>
            <Box px="10" flexDirection="column">
              <ImageModalField data={imageModalMetadata.text_prompts} header="Prompt:" />
              <ImageModalField
                data={imageModalMetadata.negative_prompts}
                header="Negative Prompt:"
              />

              <Divider pt="5"></Divider>
              <Flex>
                <Box flexDirection="column" width="50%">
                  <ImageModalField
                    data={`${imageModalMetadata.width}x${imageModalMetadata.height}`}
                    header="Dimensions (WxH)"
                  />
                </Box>
                <Box>
                  <ImageModalField data={imageModalMetadata.seed} header="Seed:" />
                </Box>
              </Flex>

              <Flex>
                <Box flexDirection="column" width="50%">
                  <ImageModalField data={imageModalMetadata.sampler} header="Sampler:" />
                </Box>
                <Box>
                  <ImageModalField data={imageModalMetadata.steps} header="Steps:" />
                </Box>
              </Flex>

              <Flex>
                <Box flexDirection="column" width="50%">
                  <ImageModalField
                    data={imageModalMetadata.strength}
                    header="Starting Image Strength:"
                  />
                </Box>
                <Box>
                  <ImageModalField
                    data={imageModalMetadata.cfg_scale}
                    header="Prompt Strength (CFG):"
                  />
                </Box>
              </Flex>
              <Flex>
                <Box flexDirection="column" width="50%">
                  <ImageModalField data={imageModalMetadata.clip_skip} header="Clip skip:" />
                </Box>
                <Box></Box>
              </Flex>

              <Flex>
                <Box flexDirection="column" width="50%">
                  <ImageModalField data={imageModalMetadata.ckpt} header="Model:" />
                </Box>
                <Box>
                  <ImageModalField data={imageModalMetadata.vae} header="VAE:" />
                </Box>
              </Flex>

              <Flex>
                <Box flexDirection="column" width="50%">
                  <ImageModalField data={imageModalMetadata.loras} header="Loras:" />
                </Box>
                <Box>
                  <ImageModalField data={imageModalMetadata.controlnet} header="Controlnet:" />
                </Box>
              </Flex>

              <Box pt="5" width="100%" display="flex" justifyContent="center">
                <Button
                  borderRadius="10"
                  variant="ghost"
                  colorScheme="blue"
                  fontSize="14"
                  fontWeight="normal"
                  isDisabled={!('text_prompts' in imageModalMetadata)}
                  onClick={() => {
                    window.api.copyToClipboard(JSON.stringify(imageModalMetadata, null, 2), 'text')
                    toast({ title: 'copied to clipboard' })
                  }}
                >
                  <Icon mr={2} as={GoSettings} />
                  Copy All Settings
                </Button>
              </Box>
            </Box>
          </Flex>
        </ModalBody>
        <ModalFooter>
          <Button onClick={handleClose}>Cancel</Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  )
}

export default ImageModal

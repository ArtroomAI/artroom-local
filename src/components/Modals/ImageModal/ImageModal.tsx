import React, { useState } from 'react'
import { useRecoilState } from 'recoil'
import * as atom from '../../../atoms/atoms'
import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  Flex,
  Box,
  Text,
  Button,
  Icon,
  Divider
} from '@chakra-ui/react'
import { BiImages, BiCopy } from 'react-icons/bi'
import { GoSettings } from 'react-icons/go'
import { AiFillFolderOpen } from 'react-icons/ai'
import ImageModalObj from '../../Reusable/ImageModalObj'

export interface ImageMetadata {
  text_prompts: string
  negative_prompts: string
  W: string
  H: string
  seed: string
  sampler: string
  steps: string
  strength: string
  cfg_scale: string
  ckpt: string
  vae: string
}

function ImageModal() {
  const [imageModalB64, setImageModalB64] = useRecoilState(
    atom.imageModalB64State
  )
  const [imageModalMetadata, setImageModalMetadata] = useRecoilState(
    atom.imageModalMetadataState
  )
  const [showImageModal, setShowImageModal] = useRecoilState(
    atom.showImageModalState
  )

  function handleClose() {
    setShowImageModal(false)
  }

  return (
    <Modal
      size="6xl"
      isOpen={showImageModal}
      onClose={handleClose}
      scrollBehavior="outside"
    >
      <ModalOverlay bg="blackAlpha.900" />
      <ModalContent>
        <ModalHeader></ModalHeader>
        <ModalBody>
          <Flex>
            <Box>
              <Box>
                <ImageModalObj b64={imageModalB64}></ImageModalObj>
                <Button
                  borderRadius="10"
                  variant="ghost"
                  colorScheme="blue"
                  mr={2}
                  fontSize="14"
                  fontWeight="normal"
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
                >
                  <Icon mr={2} as={AiFillFolderOpen} />
                  View in Files
                </Button>
              </Box>
            </Box>
            <Box px="10" flexDirection="column">
              <Box display="flex" alignItems="center" mt={2}>
                <Button borderRadius="10" variant="ghost" p={0} m={0} size="sm">
                  <Icon as={BiCopy} />
                </Button>
                <Text fontWeight="bold" color="white.800">
                  Prompt:
                </Text>
              </Box>
              <Text pl="8"> {imageModalMetadata.text_prompts}</Text>

              <Box display="flex" alignItems="center" mt={2}>
                <Button borderRadius="10" variant="ghost" p={0} m={0} size="sm">
                  <Icon as={BiCopy} />
                </Button>
                <Text fontWeight="bold" color="white.800">
                  Negative Prompt:
                </Text>
              </Box>
              <Text pl="8"> {imageModalMetadata.negative_prompts}</Text>

              <Divider pt="5"></Divider>
              <Flex>
                <Box flexDirection="column" width="50%">
                  <Box display="flex" alignItems="center" mt={2}>
                    <Button
                      borderRadius="10"
                      variant="ghost"
                      p={0}
                      m={0}
                      size="sm"
                    >
                      <Icon as={BiCopy} />
                    </Button>
                    <Text fontWeight="bold" color="white.800">
                      {`Dimensions (WxH)`}:
                    </Text>
                  </Box>
                  <Text pl="8">
                    {' '}
                    {`${imageModalMetadata.W}x${imageModalMetadata.H}`}
                  </Text>
                </Box>
                <Box>
                  <Box display="flex" alignItems="center" mt={2}>
                    <Button
                      borderRadius="10"
                      variant="ghost"
                      p={0}
                      m={0}
                      size="sm"
                    >
                      <Icon as={BiCopy} />
                    </Button>
                    <Text fontWeight="bold" color="white.800">
                      Seed:
                    </Text>
                  </Box>
                  <Text pl="8"> {imageModalMetadata.seed}</Text>
                </Box>
              </Flex>

              <Flex>
                <Box flexDirection="column" width="50%">
                  <Box display="flex" alignItems="center" mt={2}>
                    <Button
                      borderRadius="10"
                      variant="ghost"
                      p={0}
                      m={0}
                      size="sm"
                    >
                      <Icon as={BiCopy} />
                    </Button>
                    <Text fontWeight="bold" color="white.800">
                      Sampler:
                    </Text>
                  </Box>
                  <Text pl="8"> {imageModalMetadata.sampler}</Text>
                </Box>
                <Box>
                  <Box display="flex" alignItems="center" mt={2}>
                    <Button
                      borderRadius="10"
                      variant="ghost"
                      p={0}
                      m={0}
                      size="sm"
                    >
                      <Icon as={BiCopy} />
                    </Button>
                    <Text fontWeight="bold" color="white.800">
                      Steps:
                    </Text>
                  </Box>
                  <Text pl="8"> {imageModalMetadata.steps}</Text>
                </Box>
              </Flex>

              <Flex>
                <Box flexDirection="column" width="50%">
                  <Box display="flex" alignItems="center" mt={2}>
                    <Button
                      borderRadius="10"
                      variant="ghost"
                      p={0}
                      m={0}
                      size="sm"
                    >
                      <Icon as={BiCopy} />
                    </Button>
                    <Text fontWeight="bold" color="white.800">
                      Starting Image Strength:
                    </Text>
                  </Box>
                  <Text pl="8"> {imageModalMetadata.strength}</Text>
                </Box>
                <Box>
                  <Box display="flex" alignItems="center" mt={2}>
                    <Button
                      borderRadius="10"
                      variant="ghost"
                      p={0}
                      m={0}
                      size="sm"
                    >
                      <Icon as={BiCopy} />
                    </Button>
                    <Text fontWeight="bold" color="white.800">
                      {`Prompt Strength (CFG)`}:
                    </Text>
                  </Box>
                  <Text pl="8"> {imageModalMetadata.cfg_scale}</Text>
                </Box>
              </Flex>

              <Flex>
                <Box flexDirection="column" width="50%">
                  <Box display="flex" alignItems="center" mt={2}>
                    <Button
                      borderRadius="10"
                      variant="ghost"
                      p={0}
                      m={0}
                      size="sm"
                    >
                      <Icon as={BiCopy} />
                    </Button>
                    <Text fontWeight="bold" color="white.800">
                      Model:
                    </Text>
                  </Box>
                  <Text pl="8"> {imageModalMetadata.ckpt}</Text>
                </Box>
                <Box>
                  <Box display="flex" alignItems="center" mt={2}>
                    <Button
                      borderRadius="10"
                      variant="ghost"
                      p={0}
                      m={0}
                      size="sm"
                    >
                      <Icon as={BiCopy} />
                    </Button>
                    <Text fontWeight="bold" color="white.800">
                      VAE:
                    </Text>
                  </Box>
                  <Text pl="8"> {imageModalMetadata.vae}</Text>
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

import React from 'react';
import {
    useDisclosure,
    Modal,
    ModalOverlay,
    ModalContent,
    ModalHeader,
    ModalCloseButton,
    ModalBody,
    Table,
    IconButton,
    Image
} from '@chakra-ui/react';
import {
    BsInfoCircle
  } from 'react-icons/bs'
import QueueModalRow from './QueueModalRow';
function QueueModal(props) {
    const { isOpen, onOpen, onClose } = useDisclosure()
    const cancelRef = React.useRef()
    const {index, id, prompt, negative_prompt, steps, n_iter, H, W, cfg_scale, seed, ckpt, sampler, init_image, strength, mask, invert, lastItem} = props.props;
    return (
      <>
        <IconButton background='transparent'  onClick={onOpen} aria-label='View' icon={<BsInfoCircle/>}></IconButton>
        <Modal isOpen={isOpen} onClose={onClose} motionPreset='slideInBottom' scrollBehavior='outside' size='4xl'>
            <ModalOverlay />
            <ModalContent bg='gray.800'>
                <ModalHeader>Queue #{index}</ModalHeader>
                <ModalCloseButton />
                <ModalBody>
                    <Table variant='simple' color='#fff' size='sm'>
                        <QueueModalRow name='Prompt:' value={prompt}/>
                        <QueueModalRow name='Negative Prompt:' value={negative_prompt}/>
                        <QueueModalRow name='Number of images:' value={n_iter}/>
                        <QueueModalRow name='Model:' value={ckpt}/>
                        <QueueModalRow name='Dimensions:' value={W+'x'+H}/>
                        <QueueModalRow name='Seed:' value={seed}/>
                        <QueueModalRow name='Steps:' value={steps}/>
                        <QueueModalRow name='Sampler:' value={sampler}/>
                        <QueueModalRow name='CFG Scale:' value={cfg_scale}/>
                        {init_image.length > 0 ? <QueueModalRow name='Strength:' value={strength}/>: <></>}
                        {mask.length > 0 ? <QueueModalRow name='Mask:' value={mask}/>: <></>}
                        {mask.length > 0 ? <QueueModalRow name='Invert:' value={mask.length > 0 ? invert : ''}/>: <></>}
                        <QueueModalRow name='Image:' value={init_image.length > 0 && init_image.length < 250 ? init_image: ''}/>
                        <Image maxWidth={256} maxHeight={256} src={init_image}></Image>

                    </Table>
                </ModalBody>
            </ModalContent>
        </Modal>
      </>
    )
  }

export default QueueModal;

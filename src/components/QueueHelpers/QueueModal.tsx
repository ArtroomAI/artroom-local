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
} from 'react-icons/bs';
import QueueModalRow from './QueueModalRow';

function QueueModal (props: QueueTypeWithIndex) {
    const { isOpen, onOpen, onClose } = useDisclosure();

    return (
        <>
            <IconButton
                aria-label="View"
                background="transparent"
                icon={<BsInfoCircle />}
                onClick={onOpen} />

            <Modal
                isOpen={isOpen}
                motionPreset="slideInBottom"
                onClose={onClose}
                scrollBehavior="outside"
                size="4xl">
                <ModalOverlay />

                <ModalContent bg="gray.800">
                    <ModalHeader>
                        Queue #
                        {props.index}
                    </ModalHeader>

                    <ModalCloseButton />

                    <ModalBody>
                        <Table
                            color="#fff"
                            size="sm"
                            variant="simple">
                            <QueueModalRow
                                name="Prompt:"
                                value={props.text_prompts} />

                            <QueueModalRow
                                name="Negative Prompt:"
                                value={props.negative_prompts} />

                            <QueueModalRow
                                name="Number of images:"
                                value={props.n_iter} />

                            <QueueModalRow
                                name="Model:"
                                value={props.ckpt} />

                            <QueueModalRow
                                name="Dimensions:"
                                value={`${props.width}x${props.height}`} />

                            <QueueModalRow
                                name="Seed:"
                                value={props.seed} />

                            <QueueModalRow
                                name="Steps:"
                                value={props.steps} />

                            <QueueModalRow
                                name="Sampler:"
                                value={props.sampler} />

                            <QueueModalRow
                                name="CFG Scale:"
                                value={props.cfg_scale} />

                            {props.init_image.length > 0
                                ? <QueueModalRow
                                    name="Strength:"
                                    value={props.strength} />
                                : <></>}

                            {props.mask?.length > 0
                                ? <QueueModalRow
                                    name="Mask:"
                                    value={props.mask} />
                                : <></>}

                            <QueueModalRow
                                name="Image:"
                                value={props.init_image.length > 0 && props.init_image.length < 250
                                    ? props.init_image
                                    : ''} />

                            <Image
                                maxHeight={256}
                                maxWidth={256}
                                src={props.init_image} />

                        </Table>
                    </ModalBody>
                </ModalContent>
            </Modal>
        </>
    );
}

export default QueueModal;

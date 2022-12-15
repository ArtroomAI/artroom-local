import React from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import { FaFileImage } from 'react-icons/fa';
import {
    MenuButton,
    HStack,
    Icon,
    Text,
    Flex,
    Menu,
    Link,
    Tooltip
} from '@chakra-ui/react';
// Tour component
const Viewer = () => {
    // Tour state is the state which control the JoyRide component
    const [navSize, changeNavSize] = useRecoilState(atom.navSizeState);
    return (
        <>
            <Flex
                alignItems={navSize === 'small'
                    ? 'center'
                    : 'flex-start'}
                flexDir="column"
                mt={25}
                w="100%"
            >
                <Menu placement="right">
                    <Link
                        _hover={{ textDecor: 'none',
                            backgroundColor: '#AEC8CA' }}
                        borderRadius={8}
                        p={2.5}
                    >
                        <Tooltip
                            fontSize="md"
                            label={navSize === 'small'
                                ? 'View Images'
                                : ''}
                            placement="bottom"
                            shouldWrapChildren>
                            <MenuButton
                                bg="transparent"
                                className="viewer-link"
                                onClick={window.getImageDir}
                                width="100%">
                                <HStack>
                                    <Icon
                                        as={FaFileImage}
                                        color="#82AAAD"
                                        fontSize="xl"
                                        justify="center" />

                                    <Text
                                        align="center"
                                        display={navSize === 'small'
                                            ? 'none'
                                            : 'flex'}
                                        fontSize="m"
                                        pl={5}
                                        pr={10}>
                                        View Images
                                    </Text>
                                </HStack>
                            </MenuButton>
                        </Tooltip>
                    </Link>
                </Menu>
            </Flex>
        </>
    );
};

export default Viewer;

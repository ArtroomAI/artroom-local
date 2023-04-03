import React from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import {
    MenuButton,
    HStack,
    Text,
    Flex,
    Menu,
    Link,
    Tooltip,
    Image
} from '@chakra-ui/react';
import CivitaiLogo from '../images/civitai.png';
const Civitai = () => {
    const [navSize, changeNavSize] = useRecoilState(atom.navSizeState);
    return (
        <>
            <Flex
                alignItems={navSize === 'small'
                    ? 'center'
                    : 'flex-start'}
                flexDir="column"
                mt={15}
                onClick={window.api.openCivitai}
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
                                ? 'Get Models'
                                : ''}
                            placement="bottom"
                            shouldWrapChildren>
                            <MenuButton
                                bg="transparent"
                                className="civitai-link"
                                width="100%" >
                                <HStack>
                                    <Image
                                            color="#82AAAD"
                                            fontSize="xl"
                                            justifyContent="center"
                                            src={CivitaiLogo}
                                            width="25px" />

                                    <Text
                                        align="center"
                                        display={navSize === 'small'
                                            ? 'none'
                                            : 'flex'}
                                        fontSize="m"
                                        pl={5}
                                        pr={10}>
                                        Get Models
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

export default Civitai;

import React from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import EquilibriumLogo from '../images/equilibriumai.png';
import {
    MenuButton,
    HStack,
    Image,
    Text,
    Flex,
    Menu,
    Link,
    Tooltip
} from '@chakra-ui/react';
const EquilibriumAI = () => {
    const [navSize, changeNavSize] = useRecoilState(atom.navSizeState);
    return (
        <>
            <Flex
                alignItems={navSize === 'small'
                    ? 'center'
                    : 'flex-start'}
                flexDir="column"
                mt={15}
                onClick={window.openEquilibrium}
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
                                ? 'Learn More'
                                : ''}
                            placement="bottom"
                            shouldWrapChildren>
                            <MenuButton
                                bg="transparent"
                                className="equilibrium-link"
                                width="100%" >
                                <HStack>
                                    <Image
                                        color="#82AAAD"
                                        fontSize="xl"
                                        justify="center"
                                        src={EquilibriumLogo}
                                        width="25px" />

                                    <Text
                                        align="center"
                                        display={navSize === 'small'
                                            ? 'none'
                                            : 'flex'}
                                        fontSize="m"
                                        pl={5}
                                        pr={10}>
                                        Learn More
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

export default EquilibriumAI;

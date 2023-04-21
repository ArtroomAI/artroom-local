import React from 'react';
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

const EquilibriumAI = ({ navSize } : { navSize: 'small' | 'large' }) => {
    return (
        <Flex
            alignItems={navSize === 'small'
                ? 'center'
                : 'flex-start'}
            flexDir="column"
            mt={15}
            onClick={window.api.openEquilibrium}
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
                                    justifyContent="center"
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
    );
};

export default EquilibriumAI;

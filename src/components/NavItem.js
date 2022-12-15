import React from 'react';
import { Flex,
    Text,
    Icon,
    Link,
    Menu,
    MenuButton,
    HStack,
    Tooltip } from '@chakra-ui/react';

export default function NavItem ({ icon, title, active, navSize, linkTo, className }) {
    return (
        <Flex
            alignItems={navSize === 'small'
                ? 'center'
                : 'flex-start'}
            flexDir="column"
            fontSize="md"
            mt={15}
            w="100%"
        >
            <Menu placement="right">
                <Link
                    _hover={{ textDecor: 'none',
                        backgroundColor: '#AEC8CA' }}
                    borderRadius={8}
                    href={linkTo}
                    p={2.5}
                >
                    <Tooltip
                        fontSize="md"
                        label={navSize === 'small'
                            ? title
                            : ''}
                        placement="bottom"
                        shouldWrapChildren>
                        <MenuButton
                            bg="transparent"
                            width="100%">
                            <HStack className={className}>
                                <Icon
                                    as={icon}
                                    color={active
                                        ? '#82AAAD'
                                        : 'gray.500'}
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
                                    {title}
                                </Text>
                            </HStack>
                        </MenuButton>
                    </Tooltip>
                </Link>
            </Menu>
        </Flex>
    );
}

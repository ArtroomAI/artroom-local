import React from 'react'
import {
    Flex,
    Text,
    Icon,
    Link,
    Menu,
    MenuButton,
    HStack,
    Tooltip} from '@chakra-ui/react'

export default function NavItem({ icon, title, active, navSize, linkTo, className}) {
    return (
        <Flex
            mt={15}
            flexDir='column'
            w='100%'
            alignItems={navSize ==='small' ? 'center' : 'flex-start'}
            fontSize='md'
        >
            <Menu placement='right'>
                <Link href={linkTo}
                    p={2.5}
                    borderRadius={8}
                    _hover={{ textDecor: 'none', backgroundColor: '#AEC8CA' }}
                    >
                    <Tooltip shouldWrapChildren placement='bottom' label={navSize === 'small' ? title : ''} fontSize='md'>
                        <MenuButton bg='transparent' width='100%'>
                            <HStack className={className}>
                                <Icon justify='center' as={icon} fontSize='xl' color={active ? '#82AAAD' : 'gray.500'} />
                                <Text align='center' pl={5} pr={10} fontSize='m' display={navSize === 'small' ? 'none' : 'flex'}>{title}</Text>
                            </HStack>
                        </MenuButton>
                    </Tooltip>
                </Link>
            </Menu>
        </Flex>
    )
}
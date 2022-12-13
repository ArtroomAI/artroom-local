import React from 'react';
import { Button, Menu, MenuButton, MenuList, MenuItem, MenuDivider} from '@chakra-ui/react';
import {
  FaUser,
} from 'react-icons/fa';
const ProfileMenu = ({setLoggedIn}) => {

  return (
    <Menu>
      <MenuButton as={Button} leftIcon={<FaUser/>} olorScheme='teal' variant='outline'>
        My Profile
      </MenuButton>
      <MenuList>
        <MenuItem>
          Profile
        </MenuItem>
        <MenuItem>
          Settings
        </MenuItem>
        <MenuItem>
          Get Shards
        </MenuItem>
        <MenuDivider></MenuDivider>
        <MenuItem onClick={()=>setLoggedIn(false)}>
          Logout
        </MenuItem>
      </MenuList>
    </Menu>
  );
}

export default ProfileMenu;
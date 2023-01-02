import {
    Button,
    Divider,
    Image,
    HStack,
    Text,
    Menu,
    MenuButton,
    MenuList,
    MenuItem,
    MenuDivider
} from '@chakra-ui/react';
import {
    FaUser
} from 'react-icons/fa';
import Shards from '../images/shards.png';
import ProtectedReqManager from '../helpers/ProtectedReqManager';
import LoginPage from './Login/LoginPage';

const getProfile = (event) => {
    console.log("testign get profile");
    const ARTROOM_URL = process.env.REACT_APP_SERVER_URL;
    ProtectedReqManager.make_request(`${ARTROOM_URL}/users/me`).then(response => {
        console.log(response);
    }).catch(err => {
        console.log(err);
        return (
            <LoginPage />
        );
    });
}

const ProfileMenu = ({ setLoggedIn }) => (

    <Menu>
        <MenuButton
            as={Button}
            leftIcon={<FaUser />}
            colorScheme="teal"
            variant="outline">
            <HStack>
                <Text>
                    My Profile
                </Text>
                <Divider
                    color="white"
                    height="20px"
                    orientation="vertical" />

                <Image
                    src={Shards}
                    width="10px" />

                <Text>
                    3000
                </Text>
            </HStack>
        </MenuButton>

        <MenuList>
            <MenuItem onClick={getProfile}>
                Profile
            </MenuItem>

            <MenuItem>
                Settings
            </MenuItem>

            <MenuItem>
                Get More Shards
            </MenuItem>

            <MenuDivider />

            <MenuItem onClick={() => setLoggedIn(false)}>
                Logout
            </MenuItem>
        </MenuList>
    </Menu>
);

export default ProfileMenu;

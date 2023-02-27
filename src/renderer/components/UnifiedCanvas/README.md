# InvokeAI Unified Canvas component for ArtRoom

## Libraries that component requires:

1. `konva` (original ver: 8.3.13)
2. `lodash` (original ver: 4.17.21)
3. `react` 18+
4. `react-colorful` (original ver: 5.6.1)
5. `react-dropzone` (original ver: 14.2.2)
6. `react-hotkeys-hook` (original ver: 4.0.2)
7. `react-icons` (original ver: 4.4.0)
8. `react-konva` (original ver: 18.2.3)
9. `recoil`
10. `use-image` (original ver: 1.1.0)
11. `uuid` (original ver: 9.0.0)
12. `@chakra-ui/react` (original ver: 2.3.1)

Here "original ver" means versions of packages that this component used initially in InvokeAI. When tested on the blank electron app, I used the latest versions, and everything worked fine.

## How to use:

Because this component is one part of a big application, it has some structural leftovers from its parent, like dedicated components, utils, hooks, and styles folders. With small changes, it may be started as a complete smaller application. I think you will move some parts of it into more suitable places, like hooks to some global hooks folder, styles to global styles folders, etc.

Obviously component should be inside of `ChakraProvider` and `RecoilRoot`. Place the folder with the component in the correct place, and import it as follows:

```javascript
import { UnifiedCanvas } from '../path/component_name';
```

### **Very important to import the scss styles file located inside the styles folder (`index.scss`) into your app**

By default, it is configured to take as much window space as possible (`100vh`, `100vw`), so you probably will need to tweak styles to match your custom layout. As far as I know, the canvas component is not responsive in such a way that it can’t stretch like `div` and needs concrete width and height figures. You might need to add some size variables that change on resizing events (although, in some way, this behavior is already baked in) for custom layout, which you will probably have.

## Structure

This mainly applies to state management since the layout part is pretty straightforward. It canvas, controls for tools, some menus, and wrapper leftovers from the complete InvokeAI application. Everything else was removed.

Redux was removed from logic, but I left nearly all code (reducers, store, actions) as comments for reference and moved them to the examples folder. Originally app had 4 reducers:

- canvas (`examples/canvas.store.example/canvasSlice.example.ts`). The only one that fully transitioned to recoil (except for a few properties that were not used anywhere);
- gallery (`examples/gallery.example/store/gallerySlice.example.ts`). Entirely removed, but it had some functions that canvas needed;
- options (`examples/options.store.example/optionsSlice.example.ts`). Entirely removed;
- system (`examples/system.store.example/systemSlice.example.ts`). Using only one property;

When transitioning from redux to recoil, I tried to create some conventions:

- Each atom represents one state property from redux;
- For easy search in references, the name of the atom always consists of `name_of_redux_property` + "Atom". For example, if the property is called `isDrawing,` the atom will be called `isDrawingAtom`;
- In general, the first word in the key in an atom object says what reducer it used to be. If the key is `canvas.brushColor`, it means in canvas reducer, was state property `brushColor,` and it is recoil alternative to it. **Not applied if this is selector**;
- From what I've seen, recoil does not have first-party alternatives to redux actions when you can change a few states at one go, so to achieve this, here selectors as setters are used;
- For easy search in references, all selector actions have a name that consists of `name_of_redux_action` + "Action". For example, if the redux action was called `commitStagingAreaImage`, the selector action will be called `commitStagingAreaImageAction`;
- If the key of action has the format `name_of_reducer.action_name.action`, it is a direct recoil alternative to past redux action;

### Sockets

It looks like the original InvokeAI application communicates with the server via sockets. In the `examples/socketio.examples` folder, there is code responsible for sending-receiving data, generating results, requests, etc. It might be very useful when connecting to your implementation.

## Placeholders

Because some features rely on the server, like exporting images, uploading, and even copying to the clipboard, I needed to disable them. For some actions, an alert popup with the action name will appear. You can continue from there. Some actions are called only in sockets events and are out of action too. When dragging an image to canvas, data will appear in the console.

## Other notes

- If you are using prettier, it probably will not be happy, but I could not guess your config. Please fix possible warnings
- There is some feature of the intermediate image. I don’t know what it is doing and how to trigger it, and it relies on a gallery reducer, which is not needed here. It will not work. But all code is present for further research.
- There are 3 themes configs, and currently, there is no way to switch between them
- There are some unused functions for reference
- Tested on fresh electron project

use std::any::TypeId;
use std::borrow::Cow;
use std::cell::UnsafeCell;
use std::sync::atomic::Ordering;

use generic_array::GenericArray;

use crate::scheduler::AccessDesc;
use crate::sealed::Sealer;
use crate::system::{SysArg, SysArgGroup};
#[cfg(debug_assertions)]
use crate::util::debug::BorrowEnforcer;
use crate::world::World;

/// Metadata of a system.
#[derive(Debug)]
pub struct SysMeta {
    /// For the Rust-only side of the ECS, using a static str is fine here,
    /// but for systems coming from WebAssembly the names are loaded at runtime.
    pub(crate) name: Cow<'static, str>,
    /// The previous tick that the system was invoked in.
    pub(crate) last_ran: u32,
}

impl SysMeta {
    /// What was the last time this system has been called?
    /// 
    /// The result is in ticks.
    #[inline]
    pub fn last_ran(&self) -> u32 {
        self.last_ran
    }

    /// Returns the name of the system.
    ///
    /// This name is determined on a best-effort basis. It may not always be entirely accurate.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }
}

// `System` must implement `Sync` such that rayon can run them on other threads.
pub trait Sys: Sync {
    /// Returns the resources that this system accesses.
    fn access(&self) -> &[AccessDesc];
    /// Returns the metadata associated with this system.
    fn meta(&self) -> &SysMeta;
    /// Executes the system.
    ///
    /// # Safety
    ///
    /// The caller must manually uphold Rust's aliasing guarantees in regards to system resource access.
    unsafe fn call(&self, world: &World);
}

/// A system with associated parameters.
pub trait TypedSys<P: SysArgGroup>: Sized {
    fn into_container(self, world: &mut World) -> SysContainer<P, Self> {
        let mut name = std::any::type_name::<Self>();
        if !name.contains('{') {
            name = name.split("::").last().unwrap_or(name);
        }

        let access = P::access(world);
        let meta = SysMeta {
            last_ran: 0, // World tick starts at 1, so this system will always run every single one of its change triggers.
            name: Cow::Borrowed(name),
        };

        let state = P::init(world, &meta);

        SysContainer {
            meta: UnsafeCell::new(meta),
            system: self,
            access,
            state: UnsafeCell::new(state),

            #[cfg(debug_assertions)]
            enforcer: BorrowEnforcer::new(),
        }
    }

    /// Execute the system.
    fn call(&self, world: &World, state: &mut P::State);
}

/// Wraps the system and all of its metadata.
///
/// This is where the the states and metadata are stored.
pub struct SysContainer<P: SysArgGroup, F: TypedSys<P>> {
    /// 
    pub(crate) meta: UnsafeCell<SysMeta>,
    /// The system that can be executed. This does not have to be a regular Rust closure.
    /// Anything that implements `TypedSys` can be executed by a system container.
    pub(crate) system: F,
    /// Persistent state of the system.
    /// 
    /// This contains the state of parameters that is persisted between invocations.
    /// An example of this would be a counter keeping its count between invocations.
    pub(crate) state: UnsafeCell<P::State>,

    #[cfg(debug_assertions)]
    pub(crate) enforcer: BorrowEnforcer,

    /// List of items that this systems needs to acess.
    #[cfg(feature = "generics")]
    pub(crate) access: GenericArray<AccessDesc, P::AccessCount>,
    /// List of items that this systems needs to acess.
    #[cfg(not(feature = "generics"))]
    pub(crate) access: SmallVec<[AccessDesc; SysArg::INLINE_SIZE]>,
}

// `Send` has to be implemented manually due to the cells in the system container.
unsafe impl<P, F> Send for SysContainer<P, F>
where
    P: SysArgGroup,
    F: TypedSys<P>,
{
}

// `Sync` has to be implemented manually due to the cells in the system container.
unsafe impl<P, F> Sync for SysContainer<P, F>
where
    P: SysArgGroup,
    F: TypedSys<P>,
{
}

macro_rules! impl_system {
    ($($gen:ident),*) => {
        impl<$($gen),*, S> Sys for SysContainer<($($gen),*), S>
        where
            $($gen: SysArg),*,
            ($($gen),*): SysArgGroup,
            S: TypedSys<($($gen),*)>
        {
            #[inline]
            fn meta(&self) -> &SysMeta {
                unsafe { &*self.meta.get().cast_const() }
            }

            fn access(&self) -> &[AccessDesc] {
                &self.access
            }

            unsafe fn call(&self, world: &World) {
                #[cfg(debug_assertions)]
                let _guard = self.enforcer.write();

                // SAFETY:
                // This is safe because every system has a unique state. At the same time a system
                // can be used on only one thread at a time.
                let state = unsafe { &mut *self.state.get() };
                self.system.call(world, state);

                let meta = unsafe { &mut *self.meta.get() };
                meta.last_ran = world.current_tick;
            }
        }

        impl<$($gen),*, Sys: Fn($($gen::Output<'_>),*)> TypedSys<($($gen),*)> for Sys
        where
            $($gen: SysArg),*,
            ($($gen),*): SysArgGroup<State = ($($gen::State),*)>
        {
            #[allow(non_snake_case)]
            fn call(&self, world: &World, state: &mut <($($gen),*) as SysArgGroup>::State) {
                let ($($gen),*) = state;
                self($(
                    $gen::before_update(world, $gen)
                ),*);

                $(
                    $gen::after_update(world, $gen);
                )*
            }
        }

        impl<$($gen),*, S> IntoSys<($($gen),*)> for S
        where
            $($gen: SysArg + 'static),*,
            ($($gen),*): SysArgGroup,
            S: TypedSys<($($gen),*)>,
            S: Fn($($gen),*) + 'static
        {
            fn into_sys(self, world: &mut World) -> impl Sys + 'static {
                self.into_container(world)
            }

            fn into_boxed_sys(self, world: &mut World) -> Box<dyn Sys> {
                Box::new(self.into_container(world))
            }
        }
    }
}

// Implement `Sys` for functions with up to 10 arguments.
impl_system!(A, B);
impl_system!(A, B, C);
impl_system!(A, B, C, D);
impl_system!(A, B, C, D, E);
impl_system!(A, B, C, D, E, F);
impl_system!(A, B, C, D, E, F, G);
impl_system!(A, B, C, D, E, F, G, H);
impl_system!(A, B, C, D, E, F, G, H, I);
impl_system!(A, B, C, D, E, F, G, H, I, J);

impl<P, F> Sys for SysContainer<P, F>
where
    P: SysArg,
    F: TypedSys<P>,
{
    #[inline]
    fn meta(&self) -> &SysMeta {
        unsafe { &*self.meta.get().cast_const() }
    }

    fn access(&self) -> &[AccessDesc] {
        &self.access
    }

    unsafe fn call(&self, world: &World) {
        // SAFETY:
        // This is safe because every system has a unique state. Additionally, a system
        // can be used on only one thread at a time.
        let state = unsafe { &mut *self.state.get() };
        self.system.call(world, state);
    }
}

impl<F: Fn(P::Output<'_>) + Sync, P: SysArg> TypedSys<P> for F {
    fn call(&self, world: &World, state: &mut P::State) {
        let p = P::before_update(world, state);
        self(p);
        P::after_update(world, state);
    }
}

/// Implemented for all types that can be converted into a system.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid system",
    label = "this system has invalid system arguments",
    note = "check the system arguments of the system, are they all valid?",
    note = "examples of valid system arguments are `Query`, `Local`, `Res`, etc..."
)]
pub trait IntoSys<P> {
    /// Converts the function into a type erased [`Sys`].
    fn into_sys(self, world: &mut World) -> impl Sys + 'static;
    /// Converts the function a type erased and boxed [`Sys`]
    fn into_boxed_sys(self, world: &mut World) -> Box<dyn Sys>;
}

#[diagnostic::do_not_recommend]
impl<F, P> IntoSys<P> for F
where
    P: SysArg + 'static,
    F: Fn(P) + 'static,
    F: TypedSys<P>,
{
    // static lifetime is required to tell Rust that the returned system does not borrow the world.
    fn into_sys(self, world: &mut World) -> impl Sys + 'static {
        self.into_container(world)
    }

    fn into_boxed_sys(self, world: &mut World) -> Box<dyn Sys> {
        Box::new(self.into_container(world))
    }
}

/// Unique ID associated with a system.
/// 
/// Simply a newtype around a `u32`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct SysId(pub(crate) u32);

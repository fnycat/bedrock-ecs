use std::any::TypeId;
use std::collections::HashMap;
use std::marker::PhantomData;

use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::count_repetitions;

use crate::scheduler::{AccessDesc, AccessType, ScheduleGraph, Scheduler};
use crate::system::{IntoSys, Sys, SysArgGroup, SysContainer, SysId, SysMeta, TypedSys};
use crate::util::IterableArray;
use crate::world::World;

#[cfg(feature = "plugins")]
use crate::plugins::WasmSystem;

pub struct Chained<P: SysArgGroup, G: SystemGroup<P>> {
    group: G,
    _marker: PhantomData<P>,
}

// impl<P: SysArgGroup, G: IntoSys<P> + 'static> IntoSys<P> for Chained<P, G> {
//     fn into_boxed_sys(self, world: &mut World) -> Box<dyn Sys> {
//         self.group.into_boxed_sys(world)
//     }

//     fn into_sys(self, world: &mut World) -> impl Sys + 'static {
//         self.group.
//     }
// }


/// Group of systems that can be inserted into a schedule.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid system group",
    label = "not a valid system group",
    note = "ensure that each of the items in the tuple is a valid system",
    note = "only tuples with up to 10 elements can be used as groups"
)]
pub trait SystemGroup<P> {
    /// Inserts this bundle into the schedule builder.
    fn insert_into(self, schedule: &mut ScheduleBuilder);

    /// Inserts this bundle into the schedule builder and collects
    /// all system IDs in the given vec.
    fn insert_into_ids(self, schedule: &mut ScheduleBuilder) -> impl IterableArray<usize>;

    // fn chain<P1: SysArgGroup, G: SystemGroup<P1>>(self, after: G) -> Chained<P1, G>;
}

impl<P: SysArgGroup, S: IntoSys<P>> SystemGroup<P> for S {
    fn insert_into(self, schedule: &mut ScheduleBuilder) {
        schedule.systems.push(self.into_boxed_sys(schedule.world));
    }

    fn insert_into_ids(self, schedule: &mut ScheduleBuilder) -> impl IterableArray<usize> {
        schedule.systems.push(self.into_boxed_sys(schedule.world));
        schedule.systems.len() - 1
    }
}

macro_rules! impl_bundle {
    ($($gen:ident),*) => {
        paste::paste! {
            #[allow(unused_parens, non_snake_case)]
            #[diagnostic::do_not_recommend]
            impl<$([<$gen Func>]),*, $($gen),*> SystemGroup<($($gen),*)> for ($([<$gen Func>]),*)
            where
                $([<$gen Func>]: SystemGroup<$gen>),*
            {
                fn insert_into(self, schedule: &mut ScheduleBuilder) {
                    let ($($gen),*) = self;
                    $(
                        $gen.insert_into(schedule);
                    )*
                }

                fn insert_into_ids(self, schedule: &mut ScheduleBuilder) -> impl IterableArray<usize> {
                    let mut ids: [usize; count_repetitions!($($gen),*)] = Default::default();
                    let mut ctr = 0;

                    let ($($gen),*) = self;
                    $(
                        $gen.insert_into(schedule);
                        ids[ctr] = schedule.systems.len() - 1;
                        ctr += 1;
                    )*

                    ids
                }
            }
        }
    }
}

// impl_bundle!(A);
impl_bundle!(A, B);
impl_bundle!(A, B, C);
impl_bundle!(A, B, C, D);
impl_bundle!(A, B, C, D, E);
impl_bundle!(A, B, C, D, E, F);
impl_bundle!(A, B, C, D, E, F, G);
impl_bundle!(A, B, C, D, E, F, G, H);
impl_bundle!(A, B, C, D, E, F, G, H, I);
impl_bundle!(A, B, C, D, E, F, G, H, I, J);

/// Trait indicating that a type can be used as a schedule label.
pub trait ScheduleLabel: 'static {
    /// Human-readable name of the label.
    const NAME: &'static str;
}

pub struct SystemDescriptor<'a> {
    name: &'a str,
    /// Dependencies of the system.
    access: &'a [AccessDesc],
}

/// Type ID of a schedule label.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScheduleLabelId(pub(crate) TypeId);

impl ScheduleLabelId {
    pub const fn of<T: ScheduleLabel>() -> ScheduleLabelId {
        ScheduleLabelId(TypeId::of::<T>())
    }
}

const LABEL_MAP_STACK_SIZE: usize = 5;

pub struct ScheduleBuilder<'w> {
    /// Next system schedule ID to be assigned.
    next_id: Arc<AtomicU32>,
    world: &'w mut World,

    /// Maps labels to systems associated with that label.
    /// 
    /// A single label may map to a group of systems.
    labels: HashMap<ScheduleLabelId, SmallVec<[usize; LABEL_MAP_STACK_SIZE]>>,
    systems: Vec<Box<dyn Sys>>,
}

impl<'w> ScheduleBuilder<'w> {
    pub fn new(world: &'w mut World) -> ScheduleBuilder<'w> {
        ScheduleBuilder {
            next_id: Arc::new(AtomicU32::new(0)),
            world,

            labels: HashMap::new(),
            systems: Vec::new(),
        }
    }

    /// Adds a system or group of systems to the schedule. These systems will be unlabeled and therefore
    /// cannot be referenced.
    /// 
    /// Use [`add_labeled`] if you want to create explicit dependencies.
    /// 
    /// [`add_labeled`]: ScheduleBuilder::add_labeled
    #[must_use = "dropping the builder without calling `ScheduleBuilder::schedule` will do nothing"]
    pub fn add<G, P>(mut self, systems: G) -> ScheduleBuilder<'w>
    where
        G: SystemGroup<P>,
    {
        systems.insert_into(&mut self);

        self
    }

    /// Adds a system or group of systems to the schedule and associates a label with them.
    /// This label can later be used to define dependencies between systems.
    /// 
    /// If you don't need an explicit label, you can use the slightly more convenient [`add`] instead.
    /// 
    /// [`add`]: ScheduleBuilder::add
    #[must_use = "dropping the builder without calling `ScheduleBuilder::schedule` will do nothing"]
    pub fn add_labeled<L, G, P>(mut self, label: L, systems: G) -> ScheduleBuilder<'w> 
    where
        L: ScheduleLabel,
        G: SystemGroup<P>
    {
        let ids = systems.insert_into_ids(&mut self);
        let ids = SmallVec::from_iter(ids.array_iter());

        tracing::trace!("Created label `{}` with IDs: {ids:?}", L::NAME);

        self.labels.insert(ScheduleLabelId::of::<L>(), ids);        
        self
    }

    #[must_use = "dropping the builder without calling `ScheduleBuilder::schedule` will do nothing"]
    pub(crate) fn add_contained(&mut self, system: Box<dyn Sys>) {
        self.systems.push(system);
    }

    #[must_use = "the resulting scheduler must be used to run the schedule"]
    pub fn schedule(self) -> Scheduler {
        // Build the dependency graph
        let mut graph = ScheduleGraph::new();

        let mut writers = FxHashMap::<AccessType, usize>::default();
        let mut readers = FxHashMap::<AccessType, Vec<usize>>::default();

        graph.edges.resize_with(self.systems.len(), Vec::new);
        for (i, sys) in self.systems.iter().enumerate() {
            let access = sys.access();
            for desc in access {
                // If there is an active world system, create an edge
                if let Some(&prev_writer) = writers.get(&AccessType::World) {
                    graph.edges[prev_writer].push(i);
                }

                if desc.ty == AccessType::World {
                    // Create edge with every current reader and writer
                    for writer in writers.values() {
                        graph.edges[*writer].push(i);
                    }

                    for vec in readers.values() {
                        for reader in vec {
                            graph.edges[*reader].push(i);
                        }
                    }

                    writers.clear();
                    readers.clear();
                    writers.insert(desc.ty, i);

                    continue;
                }

                if desc.mutable {
                    // If there exist writers or readers, create an edge
                    if let Some(prev_writer) = writers.insert(desc.ty, i) {
                        graph.edges[prev_writer].push(i);
                    }

                    if let Some(prev_readers) = readers.get_mut(&desc.ty) {
                        for reader in prev_readers.iter() {
                            graph.edges[*reader].push(i);
                        }
                        prev_readers.clear();
                    }
                } else {
                    if let Some(&prev_writer) = writers.get(&desc.ty) {
                        graph.edges[prev_writer].push(i);
                    }

                    readers.entry(desc.ty).or_insert_with(Vec::new).push(i);
                }
            }
        }

        graph.edges.iter_mut().for_each(|v| v.dedup());

        // TODO: We should perform transitive reduction to remove redundant edges.

        let mut scheduler = Scheduler {
            curr_in_degrees: Vec::with_capacity(self.systems.len()),
            in_degrees: Vec::with_capacity(self.systems.len()),
            systems: self.systems,
            graph,
            ..Default::default()
        };

        scheduler.build_static_in_degrees();
        scheduler.reset_in_degrees();

        println!("{:?} {:?}", scheduler.in_degrees, scheduler.graph);

        scheduler
    }
}
